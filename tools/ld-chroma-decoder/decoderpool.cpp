/************************************************************************

    decoderpool.cpp

    ld-chroma-decoder - Colourisation filter for ld-decode
    Copyright (C) 2018-2019 Simon Inns
    Copyright (C) 2021 Phillip Blucas
    Copyright (C) 2021 Adam Sampson

    This file is part of ld-decode-tools.

    ld-chroma-decoder is free software: you can redistribute it and/or
    modify it under the terms of the GNU General Public License as
    published by the Free Software Foundation, either version 3 of the
    License, or (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

************************************************************************/

#include "decoderpool.h"

DecoderPool::DecoderPool(Decoder &_decoder, QString _inputFileName,
                         LdDecodeMetaData &_ldDecodeMetaData,
                         OutputWriter::Configuration &_outputConfig, QString _outputFileName,
                         qint32 _startFrame, qint32 _length, qint32 _maxThreads)
    : decoder(_decoder), inputFileName(_inputFileName),
      outputConfig(_outputConfig), outputFileName(_outputFileName),
      startFrame(_startFrame), length(_length), maxThreads(_maxThreads),
      abort(false), ldDecodeMetaData(_ldDecodeMetaData)
{
}

Decoder& DecoderPool::getDecoder() { return decoder; }

bool DecoderPool::process()
{
    LdDecodeMetaData::VideoParameters videoParameters = ldDecodeMetaData.getVideoParameters();

    // Configure the OutputWriter, adjusting videoParameters
    outputWriter.updateConfiguration(videoParameters, outputConfig);
    outputWriter.printOutputInfo();

    // Configure the decoder, and check that it can accept this video
    if (!decoder.configure(videoParameters)) {
        return false;
    }

    // Get the decoder's lookbehind/lookahead requirements
    decoderLookBehind = decoder.getLookBehind();
    decoderLookAhead = decoder.getLookAhead();

    // Open the source video file
    if (!sourceVideo.open(inputFileName, videoParameters.fieldWidth * videoParameters.fieldHeight)) {
        // Could not open source video file
        qInfo() << "Unable to open ld-decode video file";
        return false;
    }

    // If no startFrame parameter was specified, set the start frame to 1
    if (startFrame == -1) startFrame = 1;

    if (startFrame > ldDecodeMetaData.getNumberOfFrames()) {
        qInfo() << "Specified start frame is out of bounds, only" << ldDecodeMetaData.getNumberOfFrames() << "frames available";
        return false;
    }

    // If no length parameter was specified set the length to the number of available frames
    if (length == -1) {
        length = ldDecodeMetaData.getNumberOfFrames() - (startFrame - 1);
    } else {
        if (length + (startFrame - 1) > ldDecodeMetaData.getNumberOfFrames()) {
            qInfo() << "Specified length of" << length << "exceeds the number of available frames, setting to" << ldDecodeMetaData.getNumberOfFrames() - (startFrame - 1);
            length = ldDecodeMetaData.getNumberOfFrames() - (startFrame - 1);
        }
    }

    // Open the output file
    if (outputFileName == "-") {
        // No output filename, use stdout instead
        if (!targetVideo.open(stdout, QIODevice::WriteOnly)) {
            // Failed to open stdout
            qCritical() << "Could not open stdout for output";
            sourceVideo.close();
            return false;
        }
        qInfo() << "Writing output to stdout";
    } else {
        // Open output file
        targetVideo.setFileName(outputFileName);
        if (!targetVideo.open(QIODevice::WriteOnly)) {
            // Failed to open output file
            qCritical() << "Could not open" << outputFileName << "for output";
            sourceVideo.close();
            return false;
        }
    }

    // Write the stream header (if there is one)
    const QByteArray streamHeader = outputWriter.getStreamHeader();
    if (streamHeader.size() != 0 && targetVideo.write(streamHeader) == -1) {
        qCritical() << "Writing to the output video file failed";
        return false;
    }

    qInfo() << "Using" << maxThreads << "threads";
    qInfo() << "Processing from start frame #" << startFrame << "with a length of" << length << "frames";

    // Initialise processing state
    inputFrameNumber = startFrame;
    outputFrameNumber = startFrame;
    lastFrameNumber = length + (startFrame - 1);
    totalTimer.start();

    // V4: Compute chunk assignments for parallel 3D processing.
    // Each thread gets an independent range of frames with guard frames at
    // boundaries for temporal context.  Guard frames are decoded (providing
    // the 3D model's temporal depth) but their output is discarded.
    chunkMode = false;
    if (maxThreads > 1 && decoderLookAhead > 0) {
        static constexpr qint32 GUARD_FRAMES = 5;
        const qint32 numChunks = qMin(maxThreads, length);

        if (numChunks > 1) {
            chunkMode = true;
            chunks.resize(numChunks);
            nextChunkIndex.storeRelaxed(0);

            const qint32 framesPerChunk = length / numChunks;
            const qint32 remainder = length % numChunks;

            for (qint32 i = 0; i < numChunks; i++) {
                ChunkInfo &c = chunks[i];
                // Distribute remainder frames across the first 'remainder' chunks
                c.chunkStart = startFrame + i * framesPerChunk + qMin(i, remainder);
                c.chunkEnd = c.chunkStart + framesPerChunk + (i < remainder ? 1 : 0);

                // Guard frames: extend load range, clamped to file bounds
                c.loadStart = qMax(1, c.chunkStart - GUARD_FRAMES);
                c.loadEnd = qMin(lastFrameNumber + 1, c.chunkEnd + GUARD_FRAMES);
                c.guardBefore = c.chunkStart - c.loadStart;
                c.guardAfter = c.loadEnd - c.chunkEnd;
            }

            // In chunk mode, threads load their own fields — skip shared input counter
            inputFrameNumber = lastFrameNumber + 1;

            qInfo() << "V4: Chunk parallelism enabled —" << numChunks
                    << "chunks," << GUARD_FRAMES << "guard frames,"
                    << framesPerChunk << "frames/chunk";
        }
    }

    // Start a vector of filtering threads to process the video
    QVector<QThread *> threads;
    threads.resize(maxThreads);
    for (qint32 i = 0; i < maxThreads; i++) {
        threads[i] = decoder.makeThread(abort, *this);
        threads[i]->start(QThread::LowPriority);
    }

    // Wait for the workers to finish
    for (qint32 i = 0; i < maxThreads; i++) {
        threads[i]->wait();
        delete threads[i];
    }

    // Did any of the threads abort?
    if (abort) {
        sourceVideo.close();
        targetVideo.close();
        return false;
    }

    // Check we've processed all the frames, now the workers have finished
    if (outputFrameNumber != (lastFrameNumber + 1) || !pendingOutputFrames.empty()) {
        qCritical() << "Incorrect state at end of processing —"
                    << "outputFrame:" << outputFrameNumber
                    << "expected:" << (lastFrameNumber + 1)
                    << "pending:" << pendingOutputFrames.size();
        sourceVideo.close();
        targetVideo.close();
        return false;
    }

    double totalSecs = (static_cast<double>(totalTimer.elapsed()) / 1000.0);
    qInfo() << "Processing complete -" << length << "frames in" << totalSecs << "seconds (" <<
               length / totalSecs << "FPS )";

    // Close the source video
    sourceVideo.close();

    // Close the target video
    targetVideo.close();

    return true;
}

bool DecoderPool::getInputFrames(qint32 &startFrameNumber, QVector<SourceField> &fields, qint32 &startIndex, qint32 &endIndex)
{
    QMutexLocker locker(&inputMutex);

    // Work out a reasonable batch size to provide work for all threads.
    // This assumes that the synchronisation to get a new batch is less
    // expensive than computing a single frame, so a batch size of 1 is
    // reasonable.
    const qint32 maxBatchSize = qMin(DEFAULT_BATCH_SIZE, qMax(1, length / maxThreads));

    // Work out how many frames will be in this batch
    qint32 batchFrames = qMin(maxBatchSize, lastFrameNumber + 1 - inputFrameNumber);
    if (batchFrames == 0) {
        // No more input frames
        return false;
    }

    // Advance the frame number
    startFrameNumber = inputFrameNumber;
    inputFrameNumber += batchFrames;

    // Load the fields
    SourceField::loadFields(sourceVideo, ldDecodeMetaData,
                            startFrameNumber, batchFrames, decoderLookBehind, decoderLookAhead,
                            fields, startIndex, endIndex);

    return true;
}

bool DecoderPool::putOutputFrames(qint32 startFrameNumber, const QVector<OutputFrame> &outputFrames)
{
    QMutexLocker locker(&outputMutex);

    for (qint32 i = 0; i < outputFrames.size(); i++) {
        if (!putOutputFrame(startFrameNumber + i, outputFrames[i])) {
            return false;
        }
    }

    return true;
}

bool DecoderPool::getChunkAssignment(ChunkInfo &chunk)
{
    const int idx = nextChunkIndex.fetchAndAddRelaxed(1);
    if (idx >= chunks.size()) return false;
    chunk = chunks[idx];
    return true;
}

void DecoderPool::loadFieldsSafe(SourceVideo &sourceVideo, qint32 firstFrame, qint32 numFrames,
                                 qint32 lookBehind, qint32 lookAhead,
                                 QVector<SourceField> &fields, qint32 &startIndex, qint32 &endIndex)
{
    // Two-phase field loading:
    //   Phase 1 (under mutex): read field numbers + metadata from LdDecodeMetaData
    //            — fast, just struct copies / integer lookups.
    //   Phase 2 (no mutex):    read pixel data from the thread's own SourceVideo
    //            — slow I/O, runs in parallel across all chunk threads.

    startIndex = 2 * lookBehind;
    endIndex = startIndex + (2 * numFrames);
    const qint32 totalFields = endIndex + (2 * lookAhead);
    fields.resize(totalFields);
    const qint32 numFieldPairs = totalFields / 2;

    // Per-frame metadata gathered under the lock
    struct FieldMeta {
        qint32 firstFieldNum;
        qint32 secondFieldNum;
        LdDecodeMetaData::Field firstField;
        LdDecodeMetaData::Field secondField;
        bool blank;
    };
    QVector<FieldMeta> meta(numFieldPairs);

    quint16 black;

    // ---- Phase 1: metadata (serialized, fast) ----
    {
        QMutexLocker locker(&inputMutex);
        const auto &vp = ldDecodeMetaData.getVideoParameters();
        black = vp.black16bIre;
        const qint32 numInputFrames = ldDecodeMetaData.getNumberOfFrames();
        qint32 frameNumber = firstFrame - lookBehind;

        for (qint32 i = 0; i < numFieldPairs; i++) {
            const bool useBlank = frameNumber < 1 || frameNumber > numInputFrames;
            const qint32 lookupFrame = useBlank ? 1 : frameNumber;

            meta[i].firstFieldNum  = ldDecodeMetaData.getFirstFieldNumber(lookupFrame);
            meta[i].secondFieldNum = ldDecodeMetaData.getSecondFieldNumber(lookupFrame);
            meta[i].firstField     = ldDecodeMetaData.getField(meta[i].firstFieldNum);
            meta[i].secondField    = ldDecodeMetaData.getField(meta[i].secondFieldNum);
            meta[i].blank          = useBlank;
            frameNumber++;
        }
    }
    // ---- Mutex released — Phase 2 runs lock-free ----

    const qint32 fieldLength = sourceVideo.getFieldLength();

    for (qint32 i = 0; i < numFieldPairs; i++) {
        const qint32 fi = i * 2;
        fields[fi].field     = meta[i].firstField;
        fields[fi + 1].field = meta[i].secondField;

        if (meta[i].blank) {
            fields[fi].data.fill(black, fieldLength);
            fields[fi + 1].data.fill(black, fieldLength);
        } else {
            fields[fi].data     = sourceVideo.getVideoField(meta[i].firstFieldNum);
            fields[fi + 1].data = sourceVideo.getVideoField(meta[i].secondFieldNum);
        }
    }
}

// Write one output frame. You must hold outputMutex to call this.
//
// The worker threads will complete frames in an arbitrary order, so we can't
// just write the frames to the output file directly. Instead, we keep a map of
// frames that haven't yet been written; when a new frame comes in, we check
// whether we can now write some of them out.
//
// Returns true on success, false on failure.
bool DecoderPool::putOutputFrame(qint32 frameNumber, const OutputFrame &outputFrame)
{
    // Put this frame into the map
    pendingOutputFrames[frameNumber] = outputFrame;

    // Write out as many frames as possible
    while (pendingOutputFrames.contains(outputFrameNumber)) {
        const OutputFrame& outputData = pendingOutputFrames.value(outputFrameNumber);

        // Write the frame header (if there is one)
        const QByteArray frameHeader = outputWriter.getFrameHeader();
        if (frameHeader.size() != 0 && targetVideo.write(frameHeader) == -1) {
            qCritical() << "Writing to the output video file failed";
            return false;
        }

        // Write the frame data
        if (targetVideo.write(reinterpret_cast<const char *>(outputData.data()), outputData.size() * 2) == -1) {
            qCritical() << "Writing to the output video file failed";
            return false;
        }

        pendingOutputFrames.remove(outputFrameNumber);
        outputFrameNumber++;

        const qint32 outputCount = outputFrameNumber - startFrame;
        if ((outputCount % 32) == 0) {
            // Show an update to the user
            double fps = outputCount / (static_cast<double>(totalTimer.elapsed()) / 1000.0);
            qInfo() << outputCount << "frames processed -" << fps << "FPS";
        }
    }

    return true;
}
