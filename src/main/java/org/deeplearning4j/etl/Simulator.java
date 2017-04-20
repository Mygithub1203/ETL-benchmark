package org.deeplearning4j.etl;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.ParameterException;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.datasets.iterator.AsyncDataSetIterator;
import org.deeplearning4j.etl.tools.Modeller;
import org.deeplearning4j.etl.tools.SleepingDataSetIterator;
import org.deeplearning4j.etl.tools.SleepingTrainerContext;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.parallelism.ParallelWrapper;
import org.nd4j.linalg.dataset.DataSet;

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

        SleepingDataSetIterator iterator = new SleepingDataSetIterator(datasetTimeMillis, totalExamples);
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
    }

    public static void main(String[] args) throws Exception {
        new Simulator().run(args);
    }
}
