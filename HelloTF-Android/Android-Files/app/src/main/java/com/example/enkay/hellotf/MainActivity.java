package com.example.enkay.hellotf;
import android.content.Context;

import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.view.View;
import android.view.inputmethod.InputMethodManager;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.TextView;

import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

import java.util.Arrays;

public class MainActivity extends AppCompatActivity {

    private static final String MODEL_FILE = "file:///android_asset/frozen_iris.pb";

    private static final String INPUT_NODE = "x";
    private static final String OUTPUT_NODE = "output";

    private static final int[] INPUT_SIZE = {1,4};



    private TensorFlowInferenceInterface inferenceInterface;

    static {
        System.loadLibrary("tensorflow_inference");
    }
    public static int argmax (float [] elems)
    {
        int bestIdx = -1;
        float max = -1000;
        for (int i = 0; i < elems.length; i++) {
            float elem = elems[i];
            if (elem > max) {
                max = elem;
                bestIdx = i;
            }
        }
        return bestIdx;
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        inferenceInterface = new TensorFlowInferenceInterface();
        inferenceInterface.initializeTensorFlow(getAssets(), MODEL_FILE);
        System.out.println("model loaded successfully");
        ImageView image= (ImageView) findViewById(R.id.img);
        image.setImageResource(R.drawable.collage);


        Button button = (Button) findViewById(R.id.button);
        button.setOnClickListener(new View.OnClickListener()
        {
            public void onClick(View v) {

                InputMethodManager inputManager = (InputMethodManager) getSystemService(Context.INPUT_METHOD_SERVICE);

                inputManager.hideSoftInputFromWindow(getCurrentFocus().getWindowToken(), InputMethodManager.HIDE_NOT_ALWAYS);

                final EditText Num1 = (EditText) findViewById(R.id.f1);
                final EditText Num2 = (EditText) findViewById(R.id.f2);
                final EditText Num3 = (EditText) findViewById(R.id.f3);
                final EditText Num4 = (EditText) findViewById(R.id.f4);

                float num1 = Float.parseFloat(Num1.getText().toString());
                float num2 = Float.parseFloat(Num2.getText().toString());
                float num3 = Float.parseFloat(Num3.getText().toString());
                float num4 = Float.parseFloat(Num4.getText().toString());

                float[] inputFloats = {num1, num2, num3, num4};
               
                System.out.println(inputFloats.length);


                inferenceInterface.fillNodeFloat(INPUT_NODE, INPUT_SIZE, inputFloats);


                inferenceInterface.runInference(new String[] {OUTPUT_NODE});

                float[] result={0,0,0} ;
                //int[] result={0};
                inferenceInterface.readNodeFloat(OUTPUT_NODE, result);

                int class_id=argmax(result);

                ImageView image= (ImageView) findViewById(R.id.img);

                String s="class";
                if (class_id==0)
                {
                    s="The species is likely Iris-setosa";
                    image.setImageResource(R.drawable.setosa);
                }
                else if (class_id==1)
                {
                    s="The species is likely Iris-versicolor";
                    image.setImageResource(R.drawable.versicolor);
                }
                else if (class_id==2){
                    s="The species is likely Iris-virginica";
                    image.setImageResource(R.drawable.virginica);
                }


                //System.out.println(Float.toString(result[0]));

                final TextView textViewR = (TextView) findViewById(R.id.result);
                //textViewR.setText(Integer.toString(class_id));
                textViewR.setText(s);
            }
        });



    }
}
