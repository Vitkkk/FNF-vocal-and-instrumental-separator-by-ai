package com.vitkkk.fnfseparator;

import android.app.Activity;
import android.content.Intent;
import android.net.Uri;
import android.os.Bundle;
import android.provider.Settings;
import android.view.Gravity;
import android.view.View;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.ProgressBar;
import android.widget.TextView;
import android.widget.Toast;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class MainActivity extends Activity {
    private static final int PICK_MODEL=1001;
    private static final int PICK_AUDIO=1002;
    private final ExecutorService worker=Executors.newSingleThreadExecutor();
    private File modelFile;
    private Uri audioUri;
    private TextView modelText,audioText,status;
    private ProgressBar progress;
    private Button chooseAudio,run;

    @Override public void onCreate(Bundle b) {
        super.onCreate(b);
        modelFile=new File(getFilesDir(),"fnf_separator_v31.onnx");

        LinearLayout root=new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setPadding(dp(22),dp(34),dp(22),dp(22));
        root.setGravity(Gravity.CENTER_HORIZONTAL);

        TextView title=new TextView(this);
        title.setText("FNF Vocal + Instrumental Separator");
        title.setTextSize(24);
        title.setGravity(Gravity.CENTER);
        root.addView(title,new LinearLayout.LayoutParams(-1,-2));

        TextView sub=new TextView(this);
        sub.setText("V3.1 • processamento 100% local");
        sub.setTextSize(15);
        sub.setGravity(Gravity.CENTER);
        LinearLayout.LayoutParams sp=new LinearLayout.LayoutParams(-1,-2); sp.setMargins(0,dp(4),0,dp(28));
        root.addView(sub,sp);

        Button chooseModel=new Button(this);
        chooseModel.setText("Importar modelo .ONNX");
        chooseModel.setOnClickListener(v->pickModel());
        root.addView(chooseModel,new LinearLayout.LayoutParams(-1,-2));

        modelText=new TextView(this);
        refreshModelText();
        root.addView(modelText,new LinearLayout.LayoutParams(-1,-2));

        chooseAudio=new Button(this);
        chooseAudio.setText("Selecionar música");
        chooseAudio.setOnClickListener(v->pickAudio());
        LinearLayout.LayoutParams ap=new LinearLayout.LayoutParams(-1,-2); ap.setMargins(0,dp(20),0,0);
        root.addView(chooseAudio,ap);

        audioText=new TextView(this);
        audioText.setText("Nenhuma música selecionada");
        root.addView(audioText,new LinearLayout.LayoutParams(-1,-2));

        run=new Button(this);
        run.setText("Separar Vocals + Instrumental");
        run.setOnClickListener(v->startSeparation());
        LinearLayout.LayoutParams rp=new LinearLayout.LayoutParams(-1,-2); rp.setMargins(0,dp(24),0,0);
        root.addView(run,rp);

        progress=new ProgressBar(this,null,android.R.attr.progressBarStyleHorizontal);
        progress.setMax(100); progress.setProgress(0);
        LinearLayout.LayoutParams pp=new LinearLayout.LayoutParams(-1,dp(22)); pp.setMargins(0,dp(18),0,0);
        root.addView(progress,pp);

        status=new TextView(this);
        status.setText("Pronto"); status.setGravity(Gravity.CENTER); status.setTextSize(14);
        LinearLayout.LayoutParams stp=new LinearLayout.LayoutParams(-1,-2); stp.setMargins(0,dp(8),0,0);
        root.addView(status,stp);

        TextView note=new TextView(this);
        note.setText("Saídas: Music/FNF Separator/*_Vocals.wav e *_Instrumental.wav");
        note.setTextSize(12); note.setGravity(Gravity.CENTER);
        LinearLayout.LayoutParams np=new LinearLayout.LayoutParams(-1,-2); np.setMargins(0,dp(24),0,0);
        root.addView(note,np);

        setContentView(root);
    }

    private void pickModel() {
        Intent i=new Intent(Intent.ACTION_OPEN_DOCUMENT);
        i.setType("application/octet-stream");
        i.addCategory(Intent.CATEGORY_OPENABLE);
        startActivityForResult(i,PICK_MODEL);
    }

    private void pickAudio() {
        Intent i=new Intent(Intent.ACTION_OPEN_DOCUMENT);
        i.setType("audio/*");
        i.addCategory(Intent.CATEGORY_OPENABLE);
        startActivityForResult(i,PICK_AUDIO);
    }

    @Override protected void onActivityResult(int req,int result,Intent data) {
        super.onActivityResult(req,result,data);
        if (result!=RESULT_OK || data==null || data.getData()==null) return;
        Uri uri=data.getData();
        if (req==PICK_MODEL) {
            try (InputStream in=getContentResolver().openInputStream(uri);
                 FileOutputStream out=new FileOutputStream(modelFile)) {
                byte[] buf=new byte[1<<16]; int n;
                while ((n=in.read(buf))>0) out.write(buf,0,n);
                refreshModelText();
                Toast.makeText(this,"Modelo importado",Toast.LENGTH_SHORT).show();
            } catch (Exception e) {
                Toast.makeText(this,"Erro ao importar modelo: "+e.getMessage(),Toast.LENGTH_LONG).show();
            }
        } else if (req==PICK_AUDIO) {
            audioUri=uri;
            audioText.setText("Áudio selecionado: "+uri.getLastPathSegment());
        }
    }

    private void startSeparation() {
        if (!modelFile.exists()) { Toast.makeText(this,"Importe o modelo ONNX primeiro.",Toast.LENGTH_LONG).show(); return; }
        if (audioUri==null) { Toast.makeText(this,"Selecione uma música.",Toast.LENGTH_SHORT).show(); return; }
        run.setEnabled(false); chooseAudio.setEnabled(false); progress.setProgress(0); status.setText("Decodificando áudio…");
        Uri selected=audioUri;
        worker.submit(()->{
            try {
                AudioDecoder.AudioData audio=AudioDecoder.decode(this,selected);
                runOnUiThread(()->status.setText("Executando V3.1…"));
                SeparatorEngine engine=new SeparatorEngine();
                SeparatorEngine.Result r=engine.separate(audio,modelFile.getAbsolutePath(),(pct,msg)->runOnUiThread(()->{
                    progress.setProgress(pct); status.setText(msg);
                }));
                runOnUiThread(()->status.setText("Salvando WAVs…"));
                String base="FNF_Separated_"+System.currentTimeMillis();
                Uri vocals=WavWriter.write(this,base+"_Vocals.wav",r.vocalL,r.vocalR,44100);
                Uri inst=WavWriter.write(this,base+"_Instrumental.wav",r.instL,r.instR,44100);
                runOnUiThread(()->{
                    progress.setProgress(100);
                    status.setText("Concluído! Arquivos salvos em Music/FNF Separator");
                    Toast.makeText(this,"Vocals e instrumental geradas!",Toast.LENGTH_LONG).show();
                    run.setEnabled(true); chooseAudio.setEnabled(true);
                });
            } catch (Throwable e) {
                runOnUiThread(()->{
                    progress.setProgress(0);
                    status.setText("Erro: "+e.getMessage());
                    Toast.makeText(this,"Falha: "+e.getMessage(),Toast.LENGTH_LONG).show();
                    run.setEnabled(true); chooseAudio.setEnabled(true);
                });
            }
        });
    }

    private void refreshModelText() {
        modelText.setText(modelFile.exists()?"Modelo local: pronto ("+(modelFile.length()/1024)+" KB)":"Modelo local: ainda não importado");
    }

    private int dp(int v) { return Math.round(v*getResources().getDisplayMetrics().density); }

    @Override protected void onDestroy() {
        super.onDestroy(); worker.shutdownNow();
    }
}
