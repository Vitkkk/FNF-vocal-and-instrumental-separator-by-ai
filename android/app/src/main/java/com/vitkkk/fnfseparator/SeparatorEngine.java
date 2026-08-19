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
        final float[] vocalL, vocalR, instL, instR;
        Result(float[] vocalL, float[] vocalR, float[] instL, float[] instR) {
            this.vocalL=vocalL; this.vocalR=vocalR; this.instL=instL; this.instR=instR;
        }
    }

    private static final int SR=44100, NFFT=1024, HOP=512, BINS=513, CHUNK=384;
    private static final float MEAN=0.13166308403015137f;
    private static final float STD=0.244642436504364f;
    private final FloatFFT_1D fft = new FloatFFT_1D(NFFT);
    private final float[] window = new float[NFFT];
    private final float windowSum;

    SeparatorEngine() {
        float s=0f;
        for (int i=0;i<NFFT;i++) {
            window[i]=(float)(0.5 - 0.5*Math.cos(2.0*Math.PI*i/NFFT));
            s+=window[i];
        }
        windowSum=s;
    }

    Result separate(AudioDecoder.AudioData audio, String modelPath, Progress progress) throws Exception {
        int n=Math.min(audio.left.length,audio.right.length);
        int frames=(int)Math.ceil((n + NFFT) / (double)HOP);
        float[] vocalL=new float[n], vocalR=new float[n], instL=new float[n], instR=new float[n];
        float[] norm=new float[n];

        OrtEnvironment env=OrtEnvironment.getEnvironment();
        try (OrtSession.SessionOptions opts=new OrtSession.SessionOptions();
             OrtSession session=env.createSession(modelPath,opts)) {
            String inputName=session.getInputNames().iterator().next();
            int done=0;
            for (int fs=0;fs<frames;fs+=CHUNK) {
                int fc=Math.min(CHUNK,frames-fs);
                float[] input=new float[BINS*fc];

                // Build the mono magnitude features exactly in the model's expected layout [1,513,frames].
                for (int t=0;t<fc;t++) {
                    int frame=fs+t;
                    float[] spec=forwardMono(audio.left,audio.right,n,frame);
                    for (int k=0;k<BINS;k++) {
                        float re=spec[2*k], im=spec[2*k+1];
                        float mag=(float)Math.sqrt(re*re+im*im)/windowSum;
                        float x=(float)((Math.log1p(40.0*mag)-MEAN)/STD);
                        input[k*fc+t]=x;
                    }
                }

                long[] shape={1,BINS,fc};
                float[][][] vocalMask=new float[1][BINS][fc];
                float[][][] instMask=new float[1][BINS][fc];
                try (OnnxTensor tensor=OnnxTensor.createTensor(env,FloatBuffer.wrap(input),shape);
                     OrtSession.Result out=session.run(Collections.singletonMap(inputName,tensor))) {
                    Object value=out.get(0).getValue();
                    float[][][][] masks=(float[][][][])value;
                    for (int k=0;k<BINS;k++) {
                        System.arraycopy(masks[0][0][k],0,vocalMask[0][k],0,fc);
                        System.arraycopy(masks[0][1][k],0,instMask[0][k],0,fc);
                    }
                }

                // Re-run per stereo channel and overlap-add masked frames.
                for (int t=0;t<fc;t++) {
                    int frame=fs+t;
                    reconstructFrame(audio.left,n,frame,vocalMask[0],instMask[0],t,vocalL,instL,norm);
                    reconstructFrame(audio.right,n,frame,vocalMask[0],instMask[0],t,vocalR,instR,null);
                }

                done+=fc;
                int pct=Math.min(99,(int)Math.round(done*100.0/frames));
                progress.onProgress(pct,"Separando áudio… "+pct+"%");
            }
        }

        for (int i=0;i<n;i++) {
            float d=Math.max(norm[i],1e-7f);
            vocalL[i]/=d; vocalR[i]/=d; instL[i]/=d; instR[i]/=d;
        }
        normalize(vocalL,vocalR); normalize(instL,instR);
        progress.onProgress(100,"Separação concluída");
        return new Result(vocalL,vocalR,instL,instR);
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

    private void reconstructFrame(float[] source,int n,int frame,float[][] vm,float[][] im,int localFrame,
                                  float[] vocal,float[] inst,float[] norm) {
        int start=frame*HOP-NFFT/2;
        float[] raw=new float[2*NFFT];
        for (int i=0;i<NFFT;i++) {
            int p=start+i;
            raw[i]=(p>=0 && p<n)?source[p]*window[i]:0f;
        }
        fft.realForwardFull(raw);

        float[] vs=new float[2*NFFT];
        float[] is=new float[2*NFFT];
        for (int k=0;k<NFFT;k++) {
            int bin=k<=NFFT/2?k:NFFT-k;
            float mv=vm[bin][localFrame];
            float mi=im[bin][localFrame];
            vs[2*k]=raw[2*k]*mv; vs[2*k+1]=raw[2*k+1]*mv;
            is[2*k]=raw[2*k]*mi; is[2*k+1]=raw[2*k+1]*mi;
        }
        fft.complexInverse(vs,true);
        fft.complexInverse(is,true);
        for (int i=0;i<NFFT;i++) {
            int p=start+i;
            if (p<0 || p>=n) continue;
            float w=window[i];
            vocal[p]+=vs[2*i]*w;
            inst[p]+=is[2*i]*w;
            if (norm!=null) norm[p]+=w*w;
        }
    }

    private static void normalize(float[] l,float[] r) {
        float peak=0f;
        for (int i=0;i<l.length;i++) peak=Math.max(peak,Math.max(Math.abs(l[i]),Math.abs(r[i])));
        if (peak>0.99f) {
            float g=0.99f/peak;
            for (int i=0;i<l.length;i++) { l[i]*=g; r[i]*=g; }
        }
    }
}
