package org.deeplearning4j.etl.tools;

import lombok.extern.slf4j.Slf4j;
import org.apache.commons.io.FileUtils;
import org.nd4j.linalg.factory.RandomFactory;
import org.nd4j.rng.NativeRandom;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ThreadLocalRandom;

/**
 * Created by justin on 4/20/17.
 */
@Slf4j
public class CSVGenerator {

    // for CSVRecordReader
    public static int numExamples = 1;
    public static int numFeaturesX = 600;
    public static int numFeaturesY = 16000;
    public static int numLabels = 256;

    // for CSVSequenceRecordReader
//    public static int numExamples = 16000;
//    public static int numFeaturesX = 256;
//    public static int numFeaturesY = 1200;
//    public static int numLabels = 160;


    //'baseDir': Base directory for the data. Change this if you want to save the data somewhere else
    private static File baseDir = new File("src/main/resources/customcsv/");
    private static File baseTrainDir = new File(baseDir, "train");
    private static File featuresDirTrain = new File(baseTrainDir, "features");
    private static File labelsDirTrain = new File(baseTrainDir, "labels");
    private static File baseTestDir = new File(baseDir, "test");
    private static File featuresDirTest = new File(baseTestDir, "features");
    private static File labelsDirTest = new File(baseTestDir, "labels");

    public static void main(String[] args) {
        //Create directories
        baseDir.mkdir();
        baseTrainDir.mkdir();
        featuresDirTrain.mkdir();
        labelsDirTrain.mkdir();
        baseTestDir.mkdir();
        featuresDirTest.mkdir();
        labelsDirTest.mkdir();

        File example;
        File label;

        try {
            for (int i = 0; i < numExamples; i++) {
                if (i % 100 == 0) log.info("Writing feature " + i);

                example = new File(featuresDirTrain.getAbsolutePath() + "/" + i + ".csv");
                label = new File(labelsDirTrain.getAbsolutePath() + "/" + i + ".csv");

                // write features
                List<String> features = new ArrayList<>();
                for (int j = 0; j < numFeaturesY; j++) {
                    features.add(commaSeparatedFeatures(numFeaturesX));
                }
                FileUtils.writeLines(example, features);

                // write labels
                String randomLabel = Integer.toString(ThreadLocalRandom.current().nextInt(0, numLabels - 1));
                FileUtils.writeStringToFile(label, randomLabel);
            }
        } catch(IOException e) {
            log.error("Could not complete CSV generation", e);
        }
    }

    public static String commaSeparatedFeatures(int length) {
        StringBuilder result = new StringBuilder();
        for(int i = 0; i < length; i++) {
            result.append(Integer.toString(ThreadLocalRandom.current().nextInt(0, 50)));
            result.append(",");
        }
        return result.length() > 0 ? result.substring(0, result.length() - 1): "";
    }
}
