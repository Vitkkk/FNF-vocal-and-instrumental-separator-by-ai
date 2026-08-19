package com.vitkkk.fnfseparator;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;

import org.jtransforms.fft.FloatFFT_1D;

import java.nio.FloatBuffer;
import java.util.Collections;

final class SeparatorEngine {
    interface Progress { void onProgress(int percent, String status); }

    static final class Result {
        final float[] vocalL, vocalR;
        Result(float[] vocalL, float[] vocalR) {
            this.vocalL=vocalL;
            this.vocalR=vocalR;
        }
    }

    private static final int NFFT=1024, HOP=512, BINS=513, CHUNK=384;
    private static final float MEAN=0.13166308403015137f;
    private static final float STD=0.244642436504364f;
    private final FloatFFT_1D fft = new FloatFFT_1D(NFFT);
    private final float[] window = new float[NFFT];
    private final float[] olaNorm = new float[HOP];
    private final float windowSum;

    SeparatorEngine() {
        float s=0f;
        for (int i=0;i<NFFT;i++) {
            window[i]=(float)(0.5 - 0.5*Math.cos(2.0*Math.PI*i/NFFT));
            s+=window[i];
        }
        for (int p=0;p<HOP;p++) {
            float a=window[p];
            float b=window[p+HOP];
            olaNorm[p]=Math.max(1e-7f,a*a+b*b);
        }
        windowSum=s;
    }

    Result separate(AudioDecoder.AudioData audio, String modelPath, Progress progress) throws Exception {
        int n=Math.min(audio.left.length,audio.right.length);
        int frames=(int)Math.ceil((n + NFFT) / (double)HOP);

        // Keep only the vocal stem in RAM. The instrumental is written later as mix - vocal.
        // This removes two full-song float buffers compared with the previous implementation.
        float[] vocalL=new float[n];
        float[] vocalR=new float[n];

        OrtEnvironment env=OrtEnvironment.getEnvironment();
        try (OrtSession.SessionOptions opts=new OrtSession.SessionOptions();
             OrtSession session=env.createSession(modelPath,opts)) {
            String inputName=session.getInputNames().iterator().next();
            int done=0;

            for (int fs=0;fs<frames;fs+=CHUNK) {
                int fc=Math.min(CHUNK,frames-fs);
                float[] input=new float[BINS*fc];

                for (int t=0;t<fc;t++) {
                    int frame=fs+t;
                    float[] spec=forwardMono(audio.left,audio.right,n,frame);
                    for (int k=0;k<BINS;k++) {
                        float re=spec[2*k], im=spec[2*k+1];
                        float mag=(float)Math.sqrt(re*re+im*im)/windowSum;
                        input[k*fc+t]=(float)((Math.log1p(40.0*mag)-MEAN)/STD);
                    }
                }

                long[] shape={1,BINS,fc};
                float[][] vocalMask=new float[BINS][fc];
                try (OnnxTensor tensor=OnnxTensor.createTensor(env,FloatBuffer.wrap(input),shape);
                     OrtSession.Result out=session.run(Collections.singletonMap(inputName,tensor))) {
                    float[][][][] masks=(float[][][][])out.get(0).getValue();
                    for (int k=0;k<BINS;k++) {
                        System.arraycopy(masks[0][0][k],0,vocalMask[k],0,fc);
                    }
                }

                for (int t=0;t<fc;t++) {
                    int frame=fs+t;
                    reconstructVocalFrame(audio.left,n,frame,vocalMask,t,vocalL);
                    reconstructVocalFrame(audio.right,n,frame,vocalMask,t,vocalR);
                }

                done+=fc;
                int pct=Math.min(99,(int)Math.round(done*100.0/frames));
                progress.onProgress(pct,"Separando áudio… "+pct+"%");
            }
        }

        // Normalize overlap-add using the periodic Hann^2 denominator, avoiding a full-size norm array.
        for (int i=0;i<n;i++) {
            float d=olaNorm[i%HOP];
            vocalL[i]/=d;
            vocalR[i]/=d;
        }

        progress.onProgress(100,"Separação concluída");
        return new Result(vocalL,vocalR);
    }

    private float[] forwardMono(float[] l,float[] r,int n,int frame) {
        float[] buf=new float[2*NFFT];
        int start=frame*HOP-NFFT/2;
        for (int i=0;i<NFFT;i++) {
            int p=start+i;
            float x=(p>=0 && p<n)?0.5f*(l[p]+r[p]):0f;
            buf[i]=x*window[i];
        }
        fft.realForwardFull(buf);
        return buf;
    }

    private void reconstructVocalFrame(float[] source,int n,int frame,float[][] vm,int localFrame,
                                       float[] vocal) {
        int start=frame*HOP-NFFT/2;
        float[] raw=new float[2*NFFT];
        for (int i=0;i<NFFT;i++) {
            int p=start+i;
            raw[i]=(p>=0 && p<n)?source[p]*window[i]:0f;
        }
        fft.realForwardFull(raw);

        for (int k=0;k<NFFT;k++) {
            int bin=k<=NFFT/2?k:NFFT-k;
            float mv=vm[bin][localFrame];
            raw[2*k]*=mv;
            raw[2*k+1]*=mv;
        }
        fft.complexInverse(raw,true);

        for (int i=0;i<NFFT;i++) {
            int p=start+i;
            if (p<0 || p>=n) continue;
            vocal[p]+=raw[2*i]*window[i];
        }
    }
}
