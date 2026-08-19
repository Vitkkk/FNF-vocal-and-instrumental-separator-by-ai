package com.vitkkk.fnfseparator;

import android.content.Context;
import android.media.AudioFormat;
import android.media.MediaCodec;
import android.media.MediaExtractor;
import android.media.MediaFormat;
import android.net.Uri;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;

final class AudioDecoder {
    static final class AudioData {
        final float[] left;
        final float[] right;
        final int sampleRate;
        AudioData(float[] left, float[] right, int sampleRate) {
            this.left = left; this.right = right; this.sampleRate = sampleRate;
        }
    }

    private static final class FloatList {
        private float[] data = new float[1 << 18];
        private int size = 0;
        void add(float v) {
            if (size == data.length) {
                float[] n = new float[data.length * 2];
                System.arraycopy(data, 0, n, 0, data.length);
                data = n;
            }
            data[size++] = v;
        }
        float[] toArray() {
            float[] out = new float[size];
            System.arraycopy(data, 0, out, 0, size);
            return out;
        }
    }

    static AudioData decode(Context context, Uri uri) throws Exception {
        MediaExtractor extractor = new MediaExtractor();
        extractor.setDataSource(context, uri, null);
        int track = -1;
        MediaFormat format = null;
        for (int i = 0; i < extractor.getTrackCount(); i++) {
            MediaFormat f = extractor.getTrackFormat(i);
            String mime = f.getString(MediaFormat.KEY_MIME);
            if (mime != null && mime.startsWith("audio/")) { track = i; format = f; break; }
        }
        if (track < 0 || format == null) throw new IOException("Nenhuma faixa de áudio encontrada.");
        extractor.selectTrack(track);

        String mime = format.getString(MediaFormat.KEY_MIME);
        MediaCodec codec = MediaCodec.createDecoderByType(mime);
        codec.configure(format, null, null, 0);
        codec.start();

        FloatList left = new FloatList();
        FloatList right = new FloatList();
        MediaCodec.BufferInfo info = new MediaCodec.BufferInfo();
        boolean inputDone = false, outputDone = false;
        int sampleRate = format.containsKey(MediaFormat.KEY_SAMPLE_RATE) ? format.getInteger(MediaFormat.KEY_SAMPLE_RATE) : 44100;
        int channels = format.containsKey(MediaFormat.KEY_CHANNEL_COUNT) ? format.getInteger(MediaFormat.KEY_CHANNEL_COUNT) : 2;
        int pcmEncoding = AudioFormat.ENCODING_PCM_16BIT;

        try {
            while (!outputDone) {
                if (!inputDone) {
                    int inIndex = codec.dequeueInputBuffer(10000);
                    if (inIndex >= 0) {
                        ByteBuffer in = codec.getInputBuffer(inIndex);
                        if (in != null) {
                            int n = extractor.readSampleData(in, 0);
                            if (n < 0) {
                                codec.queueInputBuffer(inIndex, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM);
                                inputDone = true;
                            } else {
                                codec.queueInputBuffer(inIndex, 0, n, extractor.getSampleTime(), 0);
                                extractor.advance();
                            }
                        }
                    }
                }

                int outIndex = codec.dequeueOutputBuffer(info, 10000);
                if (outIndex == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED) {
                    MediaFormat of = codec.getOutputFormat();
                    if (of.containsKey(MediaFormat.KEY_SAMPLE_RATE)) sampleRate = of.getInteger(MediaFormat.KEY_SAMPLE_RATE);
                    if (of.containsKey(MediaFormat.KEY_CHANNEL_COUNT)) channels = of.getInteger(MediaFormat.KEY_CHANNEL_COUNT);
                    if (of.containsKey(MediaFormat.KEY_PCM_ENCODING)) pcmEncoding = of.getInteger(MediaFormat.KEY_PCM_ENCODING);
                } else if (outIndex >= 0) {
                    ByteBuffer out = codec.getOutputBuffer(outIndex);
                    if (out != null && info.size > 0) {
                        out.position(info.offset);
                        out.limit(info.offset + info.size);
                        out.order(ByteOrder.LITTLE_ENDIAN);
                        if (pcmEncoding == AudioFormat.ENCODING_PCM_FLOAT) {
                            int frames = info.size / (4 * channels);
                            for (int f = 0; f < frames; f++) {
                                float l = out.getFloat();
                                float r = channels > 1 ? out.getFloat() : l;
                                for (int c = 2; c < channels; c++) out.getFloat();
                                left.add(l); right.add(r);
                            }
                        } else {
                            int frames = info.size / (2 * channels);
                            for (int f = 0; f < frames; f++) {
                                float l = out.getShort() / 32768f;
                                float r = channels > 1 ? out.getShort() / 32768f : l;
                                for (int c = 2; c < channels; c++) out.getShort();
                                left.add(l); right.add(r);
                            }
                        }
                    }
                    codec.releaseOutputBuffer(outIndex, false);
                    if ((info.flags & MediaCodec.BUFFER_FLAG_END_OF_STREAM) != 0) outputDone = true;
                }
            }
        } finally {
            codec.stop(); codec.release(); extractor.release();
        }

        float[] l = left.toArray(), r = right.toArray();
        if (sampleRate != 44100) {
            l = resample(l, sampleRate, 44100);
            r = resample(r, sampleRate, 44100);
            sampleRate = 44100;
        }
        return new AudioData(l, r, sampleRate);
    }

    private static float[] resample(float[] in, int from, int to) {
        if (from == to) return in;
        int n = Math.max(1, (int)Math.round(in.length * (double)to / from));
        float[] out = new float[n];
        double scale = (double)from / to;
        for (int i = 0; i < n; i++) {
            double p = i * scale;
            int a = Math.min(in.length - 1, (int)p);
            int b = Math.min(in.length - 1, a + 1);
            double t = p - a;
            out[i] = (float)(in[a] * (1.0 - t) + in[b] * t);
        }
        return out;
    }
}
