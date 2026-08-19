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
                if (buf.remaining()<4) { out.write(buf.array(),0,buf.position()); buf.clear(); }
                buf.putShort(toPcm(left[i]));
                buf.putShort(toPcm(right[i]));
            }
            if (buf.position()>0) out.write(buf.array(),0,buf.position());
        } catch (IOException e) {
            cr.delete(uri,null,null);
            throw e;
        } catch (RuntimeException e) {
            cr.delete(uri,null,null);
            throw e;
        }
        return uri;
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
