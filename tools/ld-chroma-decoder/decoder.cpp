/************************************************************************

    decoder.cpp

    ld-chroma-decoder - Colourisation filter for ld-decode
    Copyright (C) 2019-2021 Adam Sampson

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

#include "decoder.h"

#include "decoderpool.h"

qint32 Decoder::getLookBehind() const
{
    return 0;
}

qint32 Decoder::getLookAhead() const
{
    return 0;
}

DecoderThread::DecoderThread(QAtomicInt& _abort, DecoderPool& _decoderPool, QObject *parent)
    : QThread(parent), abort(_abort), decoderPool(_decoderPool), outputWriter(_decoderPool.getOutputWriter())
{
}

void DecoderThread::run()
{
    // V4: If chunk mode is active, take the chunk-parallel path.
    if (decoderPool.isChunkMode()) {
        runChunk();
        return;
    }

    // --- Original sequential batch path (1D/2D, or single-thread) ---
    QVector<SourceField> inputFields;
    QVector<ComponentFrame> componentFrames;
    QVector<OutputFrame> outputFrames;

    while (!abort) {
        // Get the next batch of fields to process
        qint32 startFrameNumber, startIndex, endIndex;
        if (!decoderPool.getInputFrames(startFrameNumber, inputFields, startIndex, endIndex)) {
            // No more input frames -- exit
            break;
        }

        // Adjust the temporary arrays to the right size
        const qint32 numFrames = (endIndex - startIndex) / 2;
        componentFrames.resize(numFrames);
        outputFrames.resize(numFrames);

        // Decode the fields to component frames
        decodeFrames(inputFields, startIndex, endIndex, componentFrames);

        // Convert the component frames to the output format
        for (qint32 i = 0; i < numFrames; i++) {
            outputWriter.convert(componentFrames[i], outputFrames[i]);
        }

        // Write the frames to the output file
        if (!decoderPool.putOutputFrames(startFrameNumber, outputFrames)) {
            abort = true;
            break;
        }
    }
}

// =========================================================================
// V4: Chunk-parallel processing
//
// Each thread:
//   1. Atomically claims a chunk (disjoint frame range)
//   2. Opens its OWN SourceVideo (SourceVideo is not thread-safe)
//   3. Loads all fields for the chunk, including guard frames for temporal
//      context at boundaries (the 3D model has temporal depth 4)
//   4. Decodes the full range (guards + real frames)
//   5. Outputs only the non-guard (central) frames
//
// Guard frames are fully decoded so that split3D's temporal window is
// populated correctly at chunk boundaries.  Their output is discarded.
// =========================================================================
void DecoderThread::runChunk()
{
    DecoderPool::ChunkInfo chunk;
    if (!decoderPool.getChunkAssignment(chunk)) {
        return;  // More threads than chunks — nothing to do
    }

    // Open a private SourceVideo for this thread.
    // SourceVideo is NOT thread-safe (shared file position + output buffer),
    // so each chunk thread must have its own instance.
    const auto videoParameters = decoderPool.getMetaData().getVideoParameters();
    const qint32 fieldLength = videoParameters.fieldWidth * videoParameters.fieldHeight;

    SourceVideo sourceVideo;
    if (!sourceVideo.open(decoderPool.getInputFileName(), fieldLength)) {
        qCritical() << "V4: Thread" << QThread::currentThreadId()
                     << "— failed to open source video for chunk"
                     << chunk.chunkStart << "-" << chunk.chunkEnd;
        abort = true;
        return;
    }

    // Load all fields for the chunk (guard frames + real frames),
    // plus the decoder's own lookBehind/lookAhead for split3D's
    // temporal window at the edges of the loaded range.
    const qint32 numLoadFrames = chunk.loadEnd - chunk.loadStart;
    QVector<SourceField> fields;
    qint32 startIndex, endIndex;

    SourceField::loadFields(sourceVideo, decoderPool.getMetaData(),
                            chunk.loadStart, numLoadFrames,
                            decoderPool.getDecoderLookBehind(),
                            decoderPool.getDecoderLookAhead(),
                            fields, startIndex, endIndex);

    sourceVideo.close();

    if (abort) return;

    // Decode all frames (guard + real) through the full pipeline
    const qint32 numDecodeFrames = (endIndex - startIndex) / 2;
    QVector<ComponentFrame> componentFrames(numDecodeFrames);

    decodeFrames(fields, startIndex, endIndex, componentFrames);

    if (abort) return;

    // Convert and output only the non-guard (central) frames.
    // componentFrames[0]         = loadStart
    // componentFrames[guardBefore] = chunkStart  (first real output frame)
    const qint32 outputCount = chunk.chunkEnd - chunk.chunkStart;
    QVector<OutputFrame> outputFrames(outputCount);

    for (qint32 i = 0; i < outputCount; i++) {
        outputWriter.convert(componentFrames[chunk.guardBefore + i], outputFrames[i]);
    }

    // Write frames to the output file (the pool's pending-frame map
    // handles reordering across chunks automatically).
    if (!decoderPool.putOutputFrames(chunk.chunkStart, outputFrames)) {
        abort = true;
    }
}
