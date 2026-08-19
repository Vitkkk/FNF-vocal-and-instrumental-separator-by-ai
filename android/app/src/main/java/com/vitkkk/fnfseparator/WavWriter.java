package com.vitkkk.fnfseparator;

import android.content.ContentResolver;
import android.content.ContentValues;
import android.content.Context;
import android.net.Uri;
import android.provider.MediaStore;

import java.io.BufferedOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

final class WavWriter {
    static Uri write(Context context, String fileName, float[] left, float[] right, int sampleRate) throws IOException {
        int n=Math.min(left.length,right.length);
        float gain=computeGain(left,right,n);
        return writeInternal(context,fileName,n,sampleRate,(i,ch)->ch==0?left[i]*gain:right[i]*gain);
    }

    static Uri writeDifference(Context context, String fileName,
                               float[] sourceL, float[] sourceR,
                               float[] vocalL, float[] vocalR,
                               int sampleRate) throws IOException {
        int n=Math.min(Math.min(sourceL.length,sourceR.length),Math.min(vocalL.length,vocalR.length));
        float peak=0f;
        for (int i=0;i<n;i++) {
            peak=Math.max(peak,Math.abs(sourceL[i]-vocalL[i]));
            peak=Math.max(peak,Math.abs(sourceR[i]-vocalR[i]));
        }
        float gain=peak>0.99f?0.99f/peak:1f;
        return writeInternal(context,fileName,n,sampleRate,(i,ch)->
                ch==0?(sourceL[i]-vocalL[i])*gain:(sourceR[i]-vocalR[i])*gain);
    }

    private interface SampleProvider { float get(int index,int channel); }

    private static Uri writeInternal(Context context,String fileName,int n,int sampleRate,SampleProvider provider) throws IOException {
        ContentValues v=new ContentValues();
        v.put(MediaStore.MediaColumns.DISPLAY_NAME,fileName);
        v.put(MediaStore.MediaColumns.MIME_TYPE,"audio/wav");
        v.put(MediaStore.MediaColumns.RELATIVE_PATH,"Music/FNF Separator");
        ContentResolver cr=context.getContentResolver();
        Uri uri=cr.insert(MediaStore.Audio.Media.EXTERNAL_CONTENT_URI,v);
        if (uri==null) throw new IOException("Não foi possível criar o arquivo de saída.");
        try (OutputStream raw=cr.openOutputStream(uri); BufferedOutputStream out=new BufferedOutputStream(raw)) {
            if (raw==null) throw new IOException("Não foi possível abrir o arquivo de saída.");
            writeHeader(out,n,sampleRate);
            ByteBuffer buf=ByteBuffer.allocate(8192).order(ByteOrder.LITTLE_ENDIAN);
            for (int i=0;i<n;i++) {
                if (buf.remaining()<4) {
                    out.write(buf.array(),0,buf.position());
                    buf.clear();
                }
                buf.putShort(toPcm(provider.get(i,0)));
                buf.putShort(toPcm(provider.get(i,1)));
            }
            if (buf.position()>0) out.write(buf.array(),0,buf.position());
        } catch (IOException | RuntimeException e) {
            cr.delete(uri,null,null);
            throw e;
        }
        return uri;
    }

    private static float computeGain(float[] l,float[] r,int n) {
        float peak=0f;
        for (int i=0;i<n;i++) peak=Math.max(peak,Math.max(Math.abs(l[i]),Math.abs(r[i])));
        return peak>0.99f?0.99f/peak:1f;
    }

    private static short toPcm(float x) {
        x=Math.max(-1f,Math.min(1f,x));
        return (short)Math.round(x*32767f);
    }

    private static void writeHeader(OutputStream out,int frames,int sr) throws IOException {
        int dataBytes=frames*4;
        ByteBuffer h=ByteBuffer.allocate(44).order(ByteOrder.LITTLE_ENDIAN);
        h.put(new byte[]{'R','I','F','F'}); h.putInt(36+dataBytes); h.put(new byte[]{'W','A','V','E'});
        h.put(new byte[]{'f','m','t',' '}); h.putInt(16); h.putShort((short)1); h.putShort((short)2);
        h.putInt(sr); h.putInt(sr*4); h.putShort((short)4); h.putShort((short)16);
        h.put(new byte[]{'d','a','t','a'}); h.putInt(dataBytes);
        out.write(h.array());
    }
}
