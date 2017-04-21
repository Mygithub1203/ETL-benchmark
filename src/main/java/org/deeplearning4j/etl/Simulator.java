package org.deeplearning4j.etl;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.ParameterException;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.io.FileUtils;
import org.apache.commons.io.IOUtils;
import org.datavec.api.berkeley.Pair;
import org.datavec.api.io.filters.RandomPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.records.reader.SequenceRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVSequenceRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.api.split.NumberedFileInputSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.AsyncDataSetIterator;
import org.deeplearning4j.etl.tools.CountingDataSetIterator;
import org.deeplearning4j.etl.tools.Modeller;
import org.deeplearning4j.etl.tools.SleepingDataSetIterator;
import org.deeplearning4j.etl.tools.SleepingTrainerContext;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.parallelism.ParallelWrapper;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.util.DataTypeUtil;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.ExistingMiniBatchDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.io.File;
import java.net.URL;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.concurrent.locks.LockSupport;

/**
 * @author raver119@gmail.com
 */
@Slf4j
public class Simulator {

    @Parameter(names = {"-d"}, description = "Time required to produce one DataSet")
    private int datasetTimeMillis = 35;

    @Parameter(names = {"-w"}, description = "Number of ParallelWrapper workers")
    private int numWorkers = 2;

    @Parameter(names = {"-t"}, description = "Training time of single ParallelWrapper worker")
    private int trainingTimeMillis = 150;

    @Parameter(names = {"-e"}, description = "Total number of examples to roll through")
    private int totalExamples = 1000;


    //'baseDir': Base directory for the data. Change this if you want to save the data somewhere else
    private static File baseDir = new File("src/main/resources/customcsv/");
    private static File baseTrainDir = new File(baseDir, "train");
    private static File featuresDirTrain = new File(baseTrainDir, "features");
    private static File labelsDirTrain = new File(baseTrainDir, "labels");
    private static File baseTestDir = new File(baseDir, "test");
    private static File featuresDirTest = new File(baseTestDir, "features");
    private static File labelsDirTest = new File(baseTestDir, "labels");


    public void run(String[] args) throws Exception {
        JCommander jcmdr = new JCommander(this);
        try {
            jcmdr.parse(args);
        } catch (ParameterException e) {
            //User provides invalid input -> print the usage info
            jcmdr.usage();
            try { Thread.sleep(500); } catch (Exception e2) { }
            throw e;
        }

//        downloadUCIData();

        log.info("Setting up RecordReaders...");
        // presaved images 72x3x96x96
        DataSetIterator myIterator = new ExistingMiniBatchDataSetIterator(new File("/home/justin/Datasets/umdfaces_aligned_224_presave_train"),"presave-train-%d.bin");


//        // image record reader
//        log.info("Loading paths....");
//        Random rng = new Random(123);
//        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
//        File mainPath = new File("/home/justin/Datasets/umdfaces_aligned_224/");
//        FileSplit fileSplit = new FileSplit(mainPath, NativeImageLoader.ALLOWED_FORMATS, rng);
//        RandomPathFilter randomFilter = new RandomPathFilter(rng, NativeImageLoader.ALLOWED_FORMATS);
//        InputSplit[] split;
//        split = fileSplit.sample(randomFilter, 0.998, 0.002);
//        InputSplit trainData = split[0];
//        ImageRecordReader trainRR = new ImageRecordReader(224, 224, 3, labelMaker);
//        trainRR.initialize(trainData);
//        DataSetIterator myIterator = new RecordReaderDataSetIterator(trainRR,16,1, trainRR.getLabels().size());

//        // standard CSV recordreader
//        CSVRecordReader trainFeatures = new CSVRecordReader();
//        trainFeatures.initialize(new FileSplit(new File(featuresDirTrain.getAbsolutePath() + "/0.csv")));
//        int labelIndex = 599;
//        int numClasses = 256;
//        int batchSize = 32;
//        DataSetIterator myIterator = new RecordReaderDataSetIterator(trainFeatures,batchSize,labelIndex,numClasses);

//        SequenceRecordReader trainFeatures = new CSVSequenceRecordReader();
//        trainFeatures.initialize(new NumberedFileInputSplit(featuresDirTrain.getAbsolutePath() + "/%d.csv", 0, 15999));
//        SequenceRecordReader trainLabels = new CSVSequenceRecordReader();
//        trainLabels.initialize(new NumberedFileInputSplit(labelsDirTrain.getAbsolutePath() + "/%d.csv", 0, 15999));
//        int miniBatchSize = 16;
//        int numLabelClasses = 160;
//        DataSetIterator myIterator = new SequenceRecordReaderDataSetIterator(trainFeatures, trainLabels, miniBatchSize, numLabelClasses,
//                false, SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END);


        log.info("Running simulation...");
//        SleepingDataSetIterator iterator = new SleepingDataSetIterator(datasetTimeMillis, totalExamples);
        CountingDataSetIterator iterator = new CountingDataSetIterator(myIterator);
        if (numWorkers > 1) {
            ParallelWrapper wrapper = new ParallelWrapper.Builder<>(Modeller.getConvolutionalModel())
                    .workspaceMode(WorkspaceMode.SEPARATE)
                    .averageUpdaters(false)
                    .useLegacyAveraging(false)
                    .averagingFrequency(5)
                    .prefetchBuffer(8)
                    .workers(numWorkers)
                    .useMQ(false)
                    .trainerFactory(new SleepingTrainerContext(trainingTimeMillis))
                    .build();


            wrapper.fit(iterator);
        } else {
            AsyncDataSetIterator adsi = new AsyncDataSetIterator(iterator, 8, true);
            while (adsi.hasNext()) {
                DataSet ds = adsi.next();

                LockSupport.parkNanos(trainingTimeMillis * 1000000L);
            }
        }

        log.info("Throughput for {} workers: {} datasets/second", numWorkers, iterator.getThroughput());
        log.info("Iterator details: ");
        iterator.printOutThroughput();
    }

    public static void main(String[] args) throws Exception {
        new Simulator().run(args);
    }






    //This method downloads the data, and converts the "one time series per line" format into a suitable
    //CSV sequence format that DataVec (CsvSequenceRecordReader) and DL4J can read.
    private static void downloadUCIData() throws Exception {
        log.info("Downloading UCI....");
        if (baseDir.exists()) return;    //Data already exists, don't download it again

        String url = "https://archive.ics.uci.edu/ml/machine-learning-databases/synthetic_control-mld/synthetic_control.data";
        String data = IOUtils.toString(new URL(url));

        String[] lines = data.split("\n");

        //Create directories
        baseDir.mkdir();
        baseTrainDir.mkdir();
        featuresDirTrain.mkdir();
        labelsDirTrain.mkdir();
        baseTestDir.mkdir();
        featuresDirTest.mkdir();
        labelsDirTest.mkdir();

        int lineCount = 0;
        List<Pair<String, Integer>> contentAndLabels = new ArrayList<>();
        for (String line : lines) {
            String transposed = line.replaceAll(" +", "\n");

            //Labels: first 100 examples (lines) are label 0, second 100 examples are label 1, and so on
            contentAndLabels.add(new Pair<>(transposed, lineCount++ / 100));
        }

        //Randomize and do a train/test split:
        Collections.shuffle(contentAndLabels, new Random(12345));

        int nTrain = 450;   //75% train, 25% test
        int trainCount = 0;
        int testCount = 0;
        for (Pair<String, Integer> p : contentAndLabels) {
            //Write output in a format we can read, in the appropriate locations
            File outPathFeatures;
            File outPathLabels;
            if (trainCount < nTrain) {
                outPathFeatures = new File(featuresDirTrain, trainCount + ".csv");
                outPathLabels = new File(labelsDirTrain, trainCount + ".csv");
                trainCount++;
            } else {
                outPathFeatures = new File(featuresDirTest, testCount + ".csv");
                outPathLabels = new File(labelsDirTest, testCount + ".csv");
                testCount++;
            }

            FileUtils.writeStringToFile(outPathFeatures, p.getFirst());
            FileUtils.writeStringToFile(outPathLabels, p.getSecond().toString());
        }
        log.info("Download complete.");
    }
}
