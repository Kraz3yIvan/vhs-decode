/************************************************************************

    comb.cpp

    ld-chroma-decoder - Colourisation filter for ld-decode
    Copyright (C) 2018 Chad Page
    Copyright (C) 2018-2019 Simon Inns
    Copyright (C) 2020-2021 Adam Sampson
    Copyright (C) 2021 Phillip Blucas

    This file is part of ld-decode-tools.

    V5: Batched FFT + batched ONNX inference for maximum throughput.

************************************************************************/

#include "comb.h"
#include "framecanvas.h"
#include "deemp.h"
#include "firfilter.h"
#include <algorithm>
#include <atomic>                   // V3: for std::atomic<bool>
#include <cmath>
#include <memory>
#include <utility>
#include <QMutex>
#include <vector>
#include <QMap>
#include <QThread>                  // FIX: needed for QThread::idealThreadCount()
#include <QCoreApplication>         // FIX: needed for applicationDirPath()

#include <fftw3.h>
#include <onnxruntime_cxx_api.h>


// Indexes for the candidates considered in 3D adaptive mode
enum CandidateIndex : qint32 {
    CAND_LEFT,
    CAND_RIGHT,
    CAND_UP,
    CAND_DOWN,
    CAND_PREV_FIELD,
    CAND_NEXT_FIELD,
    CAND_PREV_FRAME,
    CAND_NEXT_FRAME,
    NUM_CANDIDATES
};

// Map colours for the candidates
static constexpr quint32 CANDIDATE_SHADES[] = {
    0xFF8080, // CAND_LEFT - red
    0xFF8080, // CAND_RIGHT - red
    0xFFFF80, // CAND_UP - yellow
    0xFFFF80, // CAND_DOWN - yellow
    0x80FF80, // CAND_PREV_FIELD - green
    0x80FF80, // CAND_NEXT_FIELD - green
    0x8080FF, // CAND_PREV_FRAME - blue
    0xFF80FF, // CAND_NEXT_FRAME - purple
};

static constexpr double sin4fsc_data[] = {1.0, 0.0, -1.0, 0.0};

constexpr double sin4fsc(const qint32 i) {
    return sin4fsc_data[i % 4];
}

constexpr double cos4fsc(const qint32 i) {
    return sin4fsc(i + 1);
}

// =============================================================================
// V3: Global ONNX Runtime state
//
// Design rationale:
//   - ONE Ort::Env per process (ORT requirement). Created once under mutex.
//   - Each worker thread gets its OWN Ort::Session via thread_local.
//     This eliminates all inference serialization — 32 threads can run
//     session->Run() truly in parallel on the GPU.
//   - Raw pointers for thread_local (not unique_ptr) because MinGW64's
//     __emutls breaks non-trivial destructors on thread exit. The "leak"
//     is harmless: QThreadPool threads live until process exit, at which
//     point the OS reclaims everything.
//   - using_cuda is std::atomic<bool> for formal correctness (written once
//     under mutex, then read from many threads).
//   - modelPath is cached as a platform-appropriate string (wstring on
//     Windows, string elsewhere) so thread_local session creation doesn't
//     need QCoreApplication (which is main-thread-only on some Qt builds).
// =============================================================================

static std::unique_ptr<Ort::Env> g_ortEnv;
static std::atomic<bool>         g_envReady{false};
static std::atomic<bool>         g_using_cuda{false};
#ifdef _WIN32
static std::wstring              g_modelPath;       // Windows: ORT requires wchar_t*
#else
static std::string               g_modelPath;       // Linux/macOS: ORT uses char*
#endif
static QMutex                    g_envMutex;         // serialises one-time env init
static QMutex                    g_fftwMutex;        // V4: FFTW3 planner is not thread-safe

// Thread-local session state.
// Raw pointers: safe with MinGW64 thread_local (trivial type).
// Sessions are intentionally never deleted — see rationale above.
static thread_local Ort::Session* tl_session      = nullptr;
static thread_local bool          tl_sessionReady  = false;


// Public methods -----------------------------------------------------------------------------------------------------

Comb::Comb()
    : configurationSet(false)
{
}

qint32 Comb::Configuration::getLookBehind() const {
    if (dimensions == 3) return 1;
    return 0;
}

qint32 Comb::Configuration::getLookAhead() const {
    if (dimensions == 3) return 2;
    return 0;
}

const Comb::Configuration &Comb::getConfiguration() const {
    return configuration;
}

void Comb::updateConfiguration(const LdDecodeMetaData::VideoParameters &_videoParameters, const Comb::Configuration &_configuration)
{
    videoParameters = _videoParameters;
    configuration = _configuration;

    if (videoParameters.fieldWidth > MAX_WIDTH) qCritical() << "Comb::Comb(): Frame width exceeds allowed maximum!";
    if (((videoParameters.fieldHeight * 2) - 1) > MAX_HEIGHT) qCritical() << "Comb::Comb(): Frame height exceeds allowed maximum!";
    if (videoParameters.activeVideoStart < 16) qCritical() << "Comb::Comb(): activeVideoStart must be > 16!";
    if (fabs((videoParameters.sampleRate / videoParameters.fSC) - 4.0) > 1.0e-6)
    {
        qCritical() << "Data is not in 4fsc sample rate, color decoding will not work properly!";
    }

    configurationSet = true;
}

// -----------------------------------------------------------------------------
// [REVISED] decodeFrames: 4-Field Block / 2-Field Step (Overlap-Add)
// -----------------------------------------------------------------------------
void Comb::decodeFrames(const QVector<SourceField> &inputFields, qint32 startIndex, qint32 endIndex,
                        QVector<ComponentFrame> &componentFrames)
{
    assert(configurationSet);
    assert((componentFrames.size() * 2) == (endIndex - startIndex));

    QMap<int, std::shared_ptr<FrameBuffer>> bufferCache;

    auto getFrameBuffer = [&](int frameIdx) -> std::shared_ptr<FrameBuffer> {
        if (bufferCache.contains(frameIdx)) {
            return bufferCache[frameIdx];
        }

        auto buf = std::make_shared<FrameBuffer>(videoParameters, configuration);
        
        int fieldIdx1 = startIndex + frameIdx * 2;
        int fieldIdx2 = fieldIdx1 + 1;

        if (fieldIdx1 >= 0 && fieldIdx2 < inputFields.size()) {
            buf->loadFields(inputFields[fieldIdx1], inputFields[fieldIdx2]);
            buf->split1D();
            buf->split2D();
        } 

        bufferCache.insert(frameIdx, buf);
        return buf;
    };

    for (qint32 fieldIndex = startIndex; fieldIndex < endIndex; fieldIndex += 2) {
        int currentFrameIdx = (fieldIndex - startIndex) / 2;
        
        auto bufCurr = getFrameBuffer(currentFrameIdx);
        auto bufNext = getFrameBuffer(currentFrameIdx + 1);
        
        if (configuration.dimensions == 3) {
            bufCurr->split3D(*bufNext, currentFrameIdx);
        }
        
        if (currentFrameIdx >= 0 && currentFrameIdx < componentFrames.size()) {
            auto buf = bufCurr;
            
            componentFrames[currentFrameIdx].init(videoParameters);
            buf->setComponentFrame(componentFrames[currentFrameIdx]);

            if (configuration.dimensions == 3) {
                buf->finalizeOLA();
            }

            if (configuration.phaseCompensation) buf->splitIQlocked();
            else { buf->splitIQ(); buf->adjustY(); }

            buf->filterIQ();
            buf->doCNR();
            buf->doYNR();
            buf->transformIQ(configuration.chromaGain, configuration.chromaPhase);

            bufferCache.remove(currentFrameIdx);
        }
    }
}

// Private methods ----------------------------------------------------------------------------------------------------

Comb::FrameBuffer::FrameBuffer(const LdDecodeMetaData::VideoParameters &videoParameters_,
                               const Configuration &configuration_)
    : videoParameters(videoParameters_), configuration(configuration_)
{
    frameHeight = ((videoParameters.fieldHeight * 2) - 1);
    irescale = (videoParameters.white16bIre - videoParameters.black16bIre) / 100;

    int safeWidth = videoParameters.fieldWidth;
    int safeHeight = videoParameters.fieldHeight * 2;
    accChroma.resize(safeHeight, std::vector<double>(safeWidth, 0.0));
    weightSum.resize(safeHeight, std::vector<double>(safeWidth, 0.0));

    int totalSamples = videoParameters.fieldWidth * frameHeight;
    rawbuffer.fill(0, totalSamples); 
}

inline qint32 Comb::FrameBuffer::getFieldID(qint32 lineNumber) const
{
    bool isFirstField = ((lineNumber % 2) == 0);
    return isFirstField ? firstFieldPhaseID : secondFieldPhaseID;
}

inline bool Comb::FrameBuffer::getLinePhase(qint32 lineNumber) const
{
    qint32 fieldID = getFieldID(lineNumber);
    bool isPositivePhaseOnEvenLines = (fieldID == 1) || (fieldID == 4);
    int fieldLine = (lineNumber / 2);
    bool isEvenLine = (fieldLine % 2) == 0;
    return isEvenLine ? isPositivePhaseOnEvenLines : !isPositivePhaseOnEvenLines;
}

void Comb::FrameBuffer::loadFields(const SourceField &firstField, const SourceField &secondField)
{
    qint32 fieldLine = 0;
    rawbuffer.clear();
    for (qint32 frameLine = 0; frameLine < frameHeight; frameLine += 2) {
        rawbuffer.append(firstField.data.mid(fieldLine * videoParameters.fieldWidth, videoParameters.fieldWidth));
        rawbuffer.append(secondField.data.mid(fieldLine * videoParameters.fieldWidth, videoParameters.fieldWidth));
        fieldLine++;
    }

    firstFieldPhaseID = firstField.field.fieldPhaseID;
    secondFieldPhaseID = secondField.field.fieldPhaseID;

    for (qint32 buf = 0; buf < 3; buf++) {
        for (qint32 y = 0; y < MAX_HEIGHT; y++) {
            for (qint32 x = 0; x < MAX_WIDTH; x++) {
                clpbuffer[buf].pixel[y][x] = 0.0;
            }
        }
    }
    componentFrame = nullptr;
}

void Comb::FrameBuffer::split1D()
{
    for (qint32 lineNumber = videoParameters.firstActiveFrameLine; lineNumber < videoParameters.lastActiveFrameLine; lineNumber++) {
        const quint16 *line = rawbuffer.data() + (lineNumber * videoParameters.fieldWidth);
        for (qint32 h = videoParameters.activeVideoStart; h < videoParameters.activeVideoEnd; h++) {
            double tc1 = (line[h] - ((line[h - 2] + line[h + 2]) / 2.0)) / 2.0;
            clpbuffer[0].pixel[lineNumber][h] = tc1;
        }
    }
}

void Comb::FrameBuffer::split2D()
{
    static constexpr double blackLine[MAX_WIDTH] = {0};

    for (qint32 lineNumber = videoParameters.firstActiveFrameLine; lineNumber < videoParameters.lastActiveFrameLine; lineNumber++) {
        const double *previousLine = blackLine;
        if (lineNumber - 2 >= videoParameters.firstActiveFrameLine) {
            previousLine = clpbuffer[0].pixel[lineNumber - 2];
        }
        const double *currentLine = clpbuffer[0].pixel[lineNumber];
        const double *nextLine = blackLine;
        if (lineNumber + 2 < videoParameters.lastActiveFrameLine) {
            nextLine = clpbuffer[0].pixel[lineNumber + 2];
        }

        for (qint32 h = videoParameters.activeVideoStart; h < videoParameters.activeVideoEnd; h++) {
            double kp, kn;
            kp  = fabs(fabs(currentLine[h]) - fabs(previousLine[h]));
            kp += fabs(fabs(currentLine[h - 1]) - fabs(previousLine[h - 1]));
            kp -= (fabs(currentLine[h]) + fabs(previousLine[h - 1])) * .10;
            kn  = fabs(fabs(currentLine[h]) - fabs(nextLine[h]));
            kn += fabs(fabs(currentLine[h - 1]) - fabs(nextLine[h - 1]));
            kn -= (fabs(currentLine[h]) + fabs(nextLine[h - 1])) * .10;

            const double kRange = 45 * irescale;
            kp = qBound(0.0, 1 - (kp / kRange), 1.0);
            kn = qBound(0.0, 1 - (kn / kRange), 1.0);

            double sc = 1.0;

            if ((kn > 0) || (kp > 0)) {
                if (kn > (3 * kp)) kp = 0;
                else if (kp > (3 * kn)) kn = 0;
                sc = (2.0 / (kn + kp));
                if (sc < 1.0) sc = 1.0;
            } else {
                if ((fabs(fabs(previousLine[h]) - fabs(nextLine[h])) - fabs((nextLine[h] + previousLine[h]) * .2)) <= 0) {
                    kn = kp = 1;
                }
            }

            double tc1;
            tc1  = ((currentLine[h] - previousLine[h]) * kp * sc);
            tc1 += ((currentLine[h] - nextLine[h]) * kn * sc);
            tc1 /= 4;

            clpbuffer[1].pixel[lineNumber][h] = tc1;
        }
    }
}

#ifndef IDX3
#define IDX3(t, y, x, Nt, Ny, Nx) ((t)*(Ny)*(Nx) + (y)*(Nx) + (x))
#endif

// =============================================================================
// V5: Batched FFT + Batched ONNX Inference
//
// Three-phase pipeline per frame pair:
//   Phase 1 — Fill all block data into a contiguous buffer (CPU)
//   Phase 2 — One batched forward FFT, one batched ONNX Run(), one batched inverse FFT
//   Phase 3 — Overlap-add accumulation from batched results (CPU)
//
// Performance gains over V3/V4:
//   - fftw_plan_many_dft:  all blocks in one FFTW call (SIMD-optimal)
//   - Batched ONNX:        one GPU kernel launch instead of ~5000
//   - FFTW_MEASURE:        optimal algorithm selection (cached across frames)
//   - Static buffers:      zero per-frame allocation overhead
//   - Pre-computed windows: avoid redundant trig in inner loops
// =============================================================================

// --- V5: Thread-local cached state for batched FFT plans and buffers ---
// Each thread gets its own buffers and plans — safe for concurrent split3D.
// Re-allocated only if block count changes (which it shouldn't between frames).
// Raw pointers for thread_local: same MinGW64 rationale as tl_session.
static thread_local fftw_complex* tl_fftInBatch  = nullptr;
static thread_local fftw_complex* tl_fftOutBatch = nullptr;
static thread_local fftw_plan     tl_batchFwd    = nullptr;
static thread_local fftw_plan     tl_batchInv    = nullptr;
static thread_local int           tl_cachedBlockCount = 0;

// V5: Pre-computed 3D window product table [Nt][Ny][Nx]
// Avoids 3 multiplies per voxel in the inner loops.
static double g_win3D[4][16][16];
static bool   g_win3DReady = false;

static void ensureWin3D()
{
    if (g_win3DReady) return;
    constexpr int Nt = 4, Ny = 16, Nx = 16;
    for (int t = 0; t < Nt; ++t) {
        double wt = sin(M_PI * (t + 0.5) / Nt);
        for (int y = 0; y < Ny; ++y) {
            double wy = sin(M_PI * (y + 0.5) / Ny);
            for (int x = 0; x < Nx; ++x) {
                double wx = sin(M_PI * (x + 0.5) / Nx);
                g_win3D[t][y][x] = wt * wy * wx;
            }
        }
    }
    g_win3DReady = true;
}

// V5: Pre-computed reflection index LUT for RefMag channel.
// Maps (t, y, x) -> flat index into the 1024-element block.
static int g_refIdx[4][16][16];
static bool g_refIdxReady = false;

static void ensureRefIdx()
{
    if (g_refIdxReady) return;
    constexpr int Nt = 4, Ny = 16, Nx = 16;
    for (int t = 0; t < Nt; ++t) {
        int ref_t = ((2 - t) % 4 + 4) % 4;
        for (int y = 0; y < Ny; ++y) {
            int ref_y = (16 - y) % 16;
            for (int x = 0; x < Nx; ++x) {
                int ref_x = ((8 - x) % 16 + 16) % 16;
                g_refIdx[t][y][x] = IDX3(ref_t, ref_y, ref_x, Nt, Ny, Nx);
            }
        }
    }
    g_refIdxReady = true;
}

struct BlockLedger {
    int y, x;
    double blockDC;
};

void Comb::FrameBuffer::split3D(FrameBuffer &nextFrame, int frameIdx)
{
    constexpr int Nx = 16, Ny = 16, Nt = 4;
    constexpr int STEP_X = 8, STEP_Y = 8;
    constexpr int BLOCK_SIZE = Nt * Ny * Nx; // 1024

    ensureWin3D();
    ensureRefIdx();

    FrameBuffer* frames[2] = { this, &nextFrame };

    const int startY = videoParameters.firstActiveFrameLine - (Ny / 2);
    const int endY   = videoParameters.lastActiveFrameLine;
    const int startX = videoParameters.activeVideoStart - (Nx / 2);
    const int endX   = videoParameters.activeVideoEnd;

    // ---- Count blocks ----
    int numBlocks = 0;
    for (int y = startY; y < endY; y += STEP_Y)
        for (int x = startX; x < endX; x += STEP_X)
            numBlocks++;

    if (numBlocks == 0) return;

    // ---- V5: ONNX session init (once per thread) ----
    if (!g_envReady.load(std::memory_order_acquire)) {
        QMutexLocker locker(&g_envMutex);
        if (!g_envReady.load(std::memory_order_relaxed)) {
            try {
                g_ortEnv = std::make_unique<Ort::Env>(
                    ORT_LOGGING_LEVEL_WARNING, "NTSC_AI");

                QString modelPathQ = QCoreApplication::applicationDirPath()
                                   + "/chroma_net.onnx";
#ifdef _WIN32
                g_modelPath = modelPathQ.toStdWString();
#else
                g_modelPath = modelPathQ.toStdString();
#endif
                g_envReady.store(true, std::memory_order_release);
                qDebug() << "AI: Ort::Env created, model path:" << modelPathQ;
            } catch (const std::exception& e) {
                qCritical() << "AI: Failed to create Ort::Env:" << e.what();
            }
        }
    }

    if (g_envReady.load(std::memory_order_acquire) && !tl_sessionReady) {
        QMutexLocker sessionLocker(&g_envMutex);
        try {
            Ort::SessionOptions session_options;
            // V5: Allow ORT to use multiple threads internally for the
            // single large batched inference call (replaces V3's 32
            // single-threaded sessions).
            session_options.SetIntraOpNumThreads(0);  // 0 = ORT picks optimal

#ifdef USE_CUDA
            try {
                OrtCUDAProviderOptions cuda_options{};
                cuda_options.device_id = 0;
                cuda_options.cudnn_conv_algo_search =
                    OrtCudnnConvAlgoSearchHeuristic;
                session_options.AppendExecutionProvider_CUDA(cuda_options);
                g_using_cuda.store(true, std::memory_order_relaxed);
                qDebug() << "AI: CUDA provider registered";
            } catch (const std::exception& cuda_err) {
                qWarning() << "AI: CUDA fallback to CPU:" << cuda_err.what();
            }
#endif

            tl_session = new Ort::Session(
                *g_ortEnv, g_modelPath.c_str(), session_options);
            tl_sessionReady = true;
            qDebug() << "AI: Session created"
                     << (g_using_cuda.load() ? "[CUDA/GPU]" : "[CPU]");
        } catch (const std::exception& e) {
            qCritical() << "AI: Failed to create session:" << e.what();
        }
    }

    // ---- V5: Ensure batched FFT plans and buffers (cached, FFTW_MEASURE) ----
    {
        QMutexLocker locker(&g_fftwMutex);
        if (tl_cachedBlockCount != numBlocks) {
            if (tl_batchFwd) fftw_destroy_plan(tl_batchFwd);
            if (tl_batchInv) fftw_destroy_plan(tl_batchInv);
            if (tl_fftInBatch)  fftw_free(tl_fftInBatch);
            if (tl_fftOutBatch) fftw_free(tl_fftOutBatch);

            tl_fftInBatch  = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * numBlocks * BLOCK_SIZE);
            tl_fftOutBatch = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * numBlocks * BLOCK_SIZE);

            int n[] = { Nt, Ny, Nx };
            tl_batchFwd = fftw_plan_many_dft(3, n, numBlocks,
                tl_fftInBatch,  nullptr, 1, BLOCK_SIZE,
                tl_fftOutBatch, nullptr, 1, BLOCK_SIZE,
                FFTW_FORWARD, FFTW_MEASURE);
            tl_batchInv = fftw_plan_many_dft(3, n, numBlocks,
                tl_fftOutBatch, nullptr, 1, BLOCK_SIZE,
                tl_fftInBatch,  nullptr, 1, BLOCK_SIZE,
                FFTW_BACKWARD, FFTW_MEASURE);

            tl_cachedBlockCount = numBlocks;
            qDebug() << "AI: V5 batched FFTW plans created for"
                     << numBlocks << "blocks (FFTW_MEASURE)";
        }
    }

    // ================================================================
    // PHASE 1: Fill all blocks into contiguous FFT buffer
    // ================================================================
    std::vector<BlockLedger> ledger;
    ledger.reserve(numBlocks);

    int b = 0;
    for (int y = startY; y < endY; y += STEP_Y) {
        for (int x = startX; x < endX; x += STEP_X) {

            // --- Compute block DC ---
            double blockDC = 0.0;
            int pixelCount = 0;

            for (int f = 0; f < 2; ++f) {
                for (int sub_t = 0; sub_t < 2; ++sub_t) {
                    int t = f * 2 + sub_t;
                    bool isOddField = (t % 2 != 0);

                    for (int dy = 0; dy < Ny; ++dy) {
                        int absY = y + dy;
                        if (absY < videoParameters.firstActiveFrameLine ||
                            absY >= videoParameters.lastActiveFrameLine) continue;
                        if ((absY % 2 != 0) != isOddField) continue;

                        const quint16 *lineData = frames[f]->rawbuffer.data()
                            + (absY * videoParameters.fieldWidth);
                        for (int dx = 0; dx < Nx; ++dx) {
                            int absX = x + dx;
                            if (absX >= videoParameters.activeVideoStart &&
                                absX < videoParameters.activeVideoEnd) {
                                blockDC += lineData[absX];
                                pixelCount++;
                            }
                        }
                    }
                }
            }
            if (pixelCount > 0) blockDC /= (double)pixelCount;
            ledger.push_back({y, x, blockDC});

            // --- Fill FFT input (DC-removed, windowed) ---
            const int batchOffset = b * BLOCK_SIZE;

            for (int t = 0; t < Nt; ++t) {
                FrameBuffer* currFrame = frames[t / 2];
                bool isOddField = (t % 2 != 0);

                for (int dy = 0; dy < Ny; ++dy) {
                    int absY = y + dy;
                    bool isYInside = (absY >= videoParameters.firstActiveFrameLine) &&
                                     (absY < videoParameters.lastActiveFrameLine);
                    bool isOddLine = (absY % 2 != 0);

                    for (int dx = 0; dx < Nx; ++dx) {
                        int absX = x + dx;
                        int idx = batchOffset + IDX3(t, dy, dx, Nt, Ny, Nx);

                        if (isYInside && (isOddLine == isOddField) &&
                            absX >= videoParameters.activeVideoStart &&
                            absX < videoParameters.activeVideoEnd) {
                            double val = currFrame->rawbuffer[absY * videoParameters.fieldWidth + absX];
                            tl_fftInBatch[idx][0] = (val - blockDC) * g_win3D[t][dy][dx];
                        } else {
                            tl_fftInBatch[idx][0] = 0.0;
                        }
                        tl_fftInBatch[idx][1] = 0.0;
                    }
                }
            }
            b++;
        }
    }

    // ================================================================
    // PHASE 2: Batched Forward FFT → Batched ONNX → Batched Inverse FFT
    // ================================================================

    // --- One call: all blocks forward-transformed ---
    fftw_execute(tl_batchFwd);

    // --- Batched ONNX inference ---
    if (tl_sessionReady) {
        // Build tensor [numBlocks, 2, 4, 16, 16]
        std::vector<int64_t> input_shape = {
            static_cast<int64_t>(numBlocks), 2,
            static_cast<int64_t>(Nt), static_cast<int64_t>(Ny), static_cast<int64_t>(Nx)
        };
        const size_t tensorElements = numBlocks * 2 * BLOCK_SIZE;
        std::vector<float> inputTensor(tensorElements);

        int ptr = 0;
        for (b = 0; b < numBlocks; ++b) {
            const int batchOffset = b * BLOCK_SIZE;

            // Channel 0: Magnitude
            for (int t = 0; t < Nt; ++t) {
                for (int yy = 0; yy < Ny; ++yy) {
                    for (int xx = 0; xx < Nx; ++xx) {
                        int idx = batchOffset + IDX3(t, yy, xx, Nt, Ny, Nx);
                        double re = tl_fftOutBatch[idx][0];
                        double im = tl_fftOutBatch[idx][1];
                        inputTensor[ptr++] = static_cast<float>(sqrt(re * re + im * im));
                    }
                }
            }

            // Channel 1: Reflected Magnitude (pre-computed LUT)
            for (int t = 0; t < Nt; ++t) {
                for (int yy = 0; yy < Ny; ++yy) {
                    for (int xx = 0; xx < Nx; ++xx) {
                        int idx_ref = batchOffset + g_refIdx[t][yy][xx];
                        double re = tl_fftOutBatch[idx_ref][0];
                        double im = tl_fftOutBatch[idx_ref][1];
                        inputTensor[ptr++] = static_cast<float>(sqrt(re * re + im * im));
                    }
                }
            }
        }

        // Single inference call for all blocks
        auto memInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value inTensor = Ort::Value::CreateTensor<float>(
            memInfo, inputTensor.data(), tensorElements,
            input_shape.data(), input_shape.size());

        const char* inputNames[]  = {"input"};
        const char* outputNames[] = {"output"};

        auto outputTensors = tl_session->Run(
            Ort::RunOptions{nullptr},
            inputNames, &inTensor, 1,
            outputNames, 1);

        // Apply mask to all blocks' FFT coefficients
        float* maskData = outputTensors[0].GetTensorMutableData<float>();
        int maskIdx = 0;
        for (b = 0; b < numBlocks; ++b) {
            const int batchOffset = b * BLOCK_SIZE;
            for (int i = 0; i < BLOCK_SIZE; ++i) {
                float gain = maskData[maskIdx++];
                tl_fftOutBatch[batchOffset + i][0] *= gain;
                tl_fftOutBatch[batchOffset + i][1] *= gain;
            }
        }
    }

    // --- One call: all blocks inverse-transformed ---
    fftw_execute(tl_batchInv);

    // ================================================================
    // PHASE 3: Overlap-Add accumulation
    // ================================================================
    const double invBlockSize = 1.0 / (double)BLOCK_SIZE;

    for (b = 0; b < numBlocks; ++b) {
        const int batchOffset = b * BLOCK_SIZE;
        const int by = ledger[b].y;
        const int bx = ledger[b].x;

        for (int t = 0; t < Nt; ++t) {
            FrameBuffer* targetFrame = frames[t / 2];
            bool isOddField = (t % 2 != 0);

            for (int dy = 0; dy < Ny; ++dy) {
                int absY = by + dy;
                if (absY < videoParameters.firstActiveFrameLine ||
                    absY >= videoParameters.lastActiveFrameLine) continue;
                if ((absY % 2 != 0) != isOddField) continue;

                for (int dx = 0; dx < Nx; ++dx) {
                    int absX = bx + dx;
                    if (absX < videoParameters.activeVideoStart ||
                        absX >= videoParameters.activeVideoEnd) continue;

                    int idx = batchOffset + IDX3(t, dy, dx, Nt, Ny, Nx);
                    double val = tl_fftInBatch[idx][0] * invBlockSize;
                    double w = g_win3D[t][dy][dx];

                    targetFrame->accChroma[absY][absX] += val * w;
                    targetFrame->weightSum[absY][absX] += w * w;
                }
            }
        }
    }
}

void Comb::FrameBuffer::finalizeOLA() {
    int writeHeight = videoParameters.lastActiveFrameLine;
    int writeWidth = videoParameters.activeVideoEnd;
    
    for (int y = videoParameters.firstActiveFrameLine; y < writeHeight; ++y) {
        for (int x = videoParameters.activeVideoStart; x < writeWidth; ++x) {
            double w = weightSum[y][x];
            
            if (w > 0.00001) {
                clpbuffer[2].pixel[y][x] = accChroma[y][x] / w;
            } else {
                clpbuffer[2].pixel[y][x] = 0.0;
            }
        }
    }
}

void Comb::FrameBuffer::getBestCandidate(qint32 lineNumber, qint32 h, const FrameBuffer &previousFrame, const FrameBuffer &nextFrame, qint32 &bestIndex, double &bestSample) const {
    Candidate candidates[8];
    static constexpr double LINE_BONUS = -2.0;
    static constexpr double FIELD_BONUS = LINE_BONUS - 2.0;
    static constexpr double FRAME_BONUS = FIELD_BONUS - 2.0;
    candidates[CAND_LEFT] = getCandidate(lineNumber, h, *this, lineNumber, h - 2, 0);
    candidates[CAND_RIGHT] = getCandidate(lineNumber, h, *this, lineNumber, h + 2, 0);
    candidates[CAND_UP] = getCandidate(lineNumber, h, *this, lineNumber - 2, h, LINE_BONUS);
    candidates[CAND_DOWN] = getCandidate(lineNumber, h, *this, lineNumber + 2, h, LINE_BONUS);
    if (getLinePhase(lineNumber) == getLinePhase(lineNumber - 1)) {
        candidates[CAND_PREV_FIELD] = getCandidate(lineNumber, h, previousFrame, lineNumber - 1, h, FIELD_BONUS);
        candidates[CAND_NEXT_FIELD] = getCandidate(lineNumber, h, *this, lineNumber + 1, h, FIELD_BONUS);
    } else {
        candidates[CAND_PREV_FIELD] = getCandidate(lineNumber, h, *this, lineNumber - 1, h, FIELD_BONUS);
        candidates[CAND_NEXT_FIELD] = getCandidate(lineNumber, h, nextFrame, lineNumber + 1, h, FIELD_BONUS);
    }
    candidates[CAND_PREV_FRAME] = getCandidate(lineNumber, h, previousFrame, lineNumber, h, FRAME_BONUS);
    candidates[CAND_NEXT_FRAME] = getCandidate(lineNumber, h, nextFrame, lineNumber, h, FRAME_BONUS);
    if (configuration.adaptive) {
        bestIndex = 0;
        for (qint32 i = 1; i < NUM_CANDIDATES; i++) if (candidates[i].penalty < candidates[bestIndex].penalty) bestIndex = i;
    } else bestIndex = CAND_PREV_FRAME;
    bestSample = candidates[bestIndex].sample;
}

Comb::FrameBuffer::Candidate Comb::FrameBuffer::getCandidate(qint32 refLineNumber, qint32 refH, const FrameBuffer &frameBuffer, qint32 lineNumber, qint32 h, double adjustPenalty) const {
    Candidate result;
    result.sample = frameBuffer.clpbuffer[0].pixel[lineNumber][h];
    if (lineNumber < videoParameters.firstActiveFrameLine || lineNumber >= videoParameters.lastActiveFrameLine) { result.penalty = 1000.0; return result; }
    const qint32 wantPhase = (2 + (getLinePhase(refLineNumber) ? 2 : 0) + refH) % 4;
    const qint32 havePhase = ((frameBuffer.getLinePhase(lineNumber) ? 2 : 0) + h) % 4;
    if (wantPhase != havePhase) { result.penalty = 1000.0; return result; }
    const quint16 *refLine = rawbuffer.data() + (refLineNumber * videoParameters.fieldWidth);
    const quint16 *candidateLine = frameBuffer.rawbuffer.data() + (lineNumber * videoParameters.fieldWidth);
    double yPenalty = 0.0;
    for (qint32 offset = -1; offset < 2; offset++) {
        const double refC = clpbuffer[1].pixel[refLineNumber][refH + offset];
        const double refY = refLine[refH + offset] - refC;
        const double candidateC = frameBuffer.clpbuffer[1].pixel[lineNumber][h + offset];
        const double candidateY = candidateLine[h + offset] - candidateC;
        yPenalty += fabs(refY - candidateY);
    }
    yPenalty = yPenalty / 3 / irescale;
    double iqPenalty = 0.0;
    for (qint32 offset = -1; offset < 2; offset++) {
        const double refC = clpbuffer[1].pixel[refLineNumber][refH + offset];
        const double candidateC = -frameBuffer.clpbuffer[1].pixel[lineNumber][h + offset];
        static constexpr double weights[] = {0.5, 1.0, 0.5};
        iqPenalty += fabs(refC - candidateC) * weights[offset + 1];
    }
    iqPenalty = (iqPenalty / 2 / irescale) * 0.28;
    result.penalty = yPenalty + iqPenalty + adjustPenalty;
    return result;
}

namespace {
    struct BurstInfo { double bsin, bcos; };
    constexpr double ROTATE_SIN = 0.5446390350150271;
    constexpr double ROTATE_COS = 0.838670567945424;
    BurstInfo detectBurst(const quint16* lineData, const LdDecodeMetaData::VideoParameters& videoParameters) {
        double bsin = 0, bcos = 0;
        for (qint32 i = videoParameters.colourBurstStart; i < videoParameters.colourBurstEnd; i++) {
            bsin += lineData[i] * sin4fsc(i); bcos += lineData[i] * cos4fsc(i);
        }
        const qint32 colourBurstLength = videoParameters.colourBurstEnd - videoParameters.colourBurstStart;
        bsin /= colourBurstLength; bcos /= colourBurstLength;
        const double burstNorm = qMax(sqrt(bsin * bsin + bcos * bcos), 130000.0 / 128);
        bsin /= burstNorm; bcos /= burstNorm;
        return {bsin, bcos};
    }
}

void Comb::FrameBuffer::splitIQlocked() {
    for (qint32 lineNumber = videoParameters.firstActiveFrameLine; lineNumber < videoParameters.lastActiveFrameLine; lineNumber++) {
        const quint16 *line = rawbuffer.data() + (lineNumber * videoParameters.fieldWidth);
        const auto info = detectBurst(line, videoParameters);
        double *Y = componentFrame->y(lineNumber);
        double *I = componentFrame->u(lineNumber);
        double *Q = componentFrame->v(lineNumber);
        for (qint32 h = videoParameters.activeVideoStart; h < videoParameters.activeVideoEnd; h++) {
            const auto val = clpbuffer[configuration.dimensions - 1].pixel[lineNumber][h];
            const auto lsin = val * sin4fsc(h) * 2;
            const auto lcos = val * cos4fsc(h) * 2;
            const auto ti = (lsin * info.bcos - lcos * info.bsin);
            const auto tq = (lsin * info.bsin + lcos * info.bcos);
            I[h + 1] = ti * ROTATE_COS - tq * -ROTATE_SIN;
            Q[h + 1] = -(ti * -ROTATE_SIN + tq * ROTATE_COS);
            Y[h] = line[h] - val;
        }
    }
}

void Comb::FrameBuffer::splitIQ() {
    for (qint32 lineNumber = videoParameters.firstActiveFrameLine; lineNumber < videoParameters.lastActiveFrameLine; lineNumber++) {
        const quint16 *line = rawbuffer.data() + (lineNumber * videoParameters.fieldWidth);
        double *Y = componentFrame->y(lineNumber);
        double *I = componentFrame->u(lineNumber);
        double *Q = componentFrame->v(lineNumber);
        bool linePhase = getLinePhase(lineNumber);
        double si = 0, sq = 0;
        for (qint32 h = videoParameters.activeVideoStart; h < videoParameters.activeVideoEnd; h++) {
            qint32 phase = h % 4;
            double cavg = clpbuffer[configuration.dimensions - 1].pixel[lineNumber][h];
            if (linePhase) cavg = -cavg;
            switch (phase) { case 0: sq = cavg; break; case 1: si = -cavg; break; case 2: sq = -cavg; break; case 3: si = cavg; break; }
            Y[h] = line[h]; I[h] = si; Q[h] = sq;
        }
    }
}

void Comb::FrameBuffer::filterIQ() {
    auto iqFilter = makeFIRFilter(c_colorlp_b);
    const int width = videoParameters.activeVideoEnd - videoParameters.activeVideoStart;
    std::vector<double> tempBuf(width);
    for (qint32 lineNumber = videoParameters.firstActiveFrameLine; lineNumber < videoParameters.lastActiveFrameLine; lineNumber++) {
        double *I = componentFrame->u(lineNumber) + videoParameters.activeVideoStart;
        double *Q = componentFrame->v(lineNumber) + videoParameters.activeVideoStart;
        iqFilter.apply(I, tempBuf.data(), width); std::copy(tempBuf.begin(), tempBuf.end(), I);
        iqFilter.apply(Q, tempBuf.data(), width); std::copy(tempBuf.begin(), tempBuf.end(), Q);
    }
}

void Comb::FrameBuffer::adjustY() {
    for (qint32 lineNumber = videoParameters.firstActiveFrameLine; lineNumber < videoParameters.lastActiveFrameLine; lineNumber++) {
        double *Y = componentFrame->y(lineNumber);
        double *I = componentFrame->u(lineNumber);
        double *Q = componentFrame->v(lineNumber);
        bool linePhase = getLinePhase(lineNumber);
        for (qint32 h = videoParameters.activeVideoStart; h < videoParameters.activeVideoEnd; h++) {
            double comp = 0; qint32 phase = h % 4;
            switch (phase) { case 0: comp = -Q[h]; break; case 1: comp = I[h]; break; case 2: comp = Q[h]; break; case 3: comp = -I[h]; break; }
            if (!linePhase) comp = -comp;
            Y[h] -= comp;
        }
    }
}

void Comb::FrameBuffer::doCNR() {
    if (configuration.cNRLevel == 0) return;
    const double nr_c = configuration.cNRLevel * irescale;
    auto iFilter(f_nrc); auto qFilter(f_nrc);
    const qint32 delay = c_nrc_b.size() / 2;
    std::vector<double> hpI(videoParameters.activeVideoEnd + delay), hpQ(videoParameters.activeVideoEnd + delay);
    for (qint32 lineNumber = videoParameters.firstActiveFrameLine; lineNumber < videoParameters.lastActiveFrameLine; lineNumber++) {
        double *I = componentFrame->u(lineNumber), *Q = componentFrame->v(lineNumber);
        for (qint32 h = videoParameters.activeVideoStart - delay; h < videoParameters.activeVideoStart; h++) { iFilter.feed(0.0); qFilter.feed(0.0); }
        for (qint32 h = videoParameters.activeVideoStart; h < videoParameters.activeVideoEnd; h++) { hpI[h] = iFilter.feed(I[h]); hpQ[h] = qFilter.feed(Q[h]); }
        for (qint32 h = videoParameters.activeVideoEnd; h < videoParameters.activeVideoEnd + delay; h++) { hpI[h] = iFilter.feed(0.0); hpQ[h] = qFilter.feed(0.0); }
        for (qint32 h = videoParameters.activeVideoStart; h < videoParameters.activeVideoEnd; h++) {
            double ai = hpI[h + delay], aq = hpQ[h + delay];
            if (fabs(ai) > nr_c) ai = (ai > 0) ? nr_c : -nr_c; if (fabs(aq) > nr_c) aq = (aq > 0) ? nr_c : -nr_c;
            I[h] -= ai; Q[h] -= aq;
        }
    }
}

void Comb::FrameBuffer::doYNR() {
    if (configuration.yNRLevel == 0) return;
    double nr_y = configuration.yNRLevel * irescale;
    auto yFilter(f_nr);
    const qint32 delay = c_nr_b.size() / 2;
    std::vector<double> hpY(videoParameters.activeVideoEnd + delay);
    for (qint32 lineNumber = videoParameters.firstActiveFrameLine; lineNumber < videoParameters.lastActiveFrameLine; lineNumber++) {
        double *Y = componentFrame->y(lineNumber);
        for (qint32 h = videoParameters.activeVideoStart - delay; h < videoParameters.activeVideoStart; h++) yFilter.feed(0.0);
        for (qint32 h = videoParameters.activeVideoStart; h < videoParameters.activeVideoEnd; h++) hpY[h] = yFilter.feed(Y[h]);
        for (qint32 h = videoParameters.activeVideoEnd; h < videoParameters.activeVideoEnd + delay; h++) hpY[h] = yFilter.feed(0.0);
        for (qint32 h = videoParameters.activeVideoStart; h < videoParameters.activeVideoEnd; h++) {
            double a = hpY[h + delay];
            if (fabs(a) > nr_y) a = (a > 0) ? nr_y : -nr_y;
            Y[h] -= a;
        }
    }
}

void Comb::FrameBuffer::transformIQ(double chromaGain, double chromaPhase) {
    const double theta = ((33 + chromaPhase) * M_PI) / 180;
    const double bp = sin(theta) * chromaGain;
    const double bq = cos(theta) * chromaGain;
    for (qint32 lineNumber = videoParameters.firstActiveFrameLine; lineNumber < videoParameters.lastActiveFrameLine; lineNumber++) {
        double *I = componentFrame->u(lineNumber);
        double *Q = componentFrame->v(lineNumber);
        for (qint32 h = videoParameters.activeVideoStart; h < videoParameters.activeVideoEnd; h++) {
            double U = (-bp * I[h]) + (bq * Q[h]);
            double V = ( bq * I[h]) + (bp * Q[h]);
            I[h] = U; Q[h] = V;
        }
    }
}

void Comb::FrameBuffer::overlayMap(const FrameBuffer &previousFrame, const FrameBuffer &nextFrame) {
    // Debug overlay omitted for brevity
}
