package org.deeplearning4j.updaters;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.ParameterException;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.etl.tools.Modeller;
import org.deeplearning4j.etl.tools.SleepingDataSetIterator;
import org.deeplearning4j.etl.tools.SleepingTrainerContext;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.parallelism.ParallelWrapper;

/**
 * @author raver119@gmail.com
 */
@Slf4j
public class StandaloneSimulator {

    private long timeETL = 10;

    private long timeFF = 50;

    private long timeBP = 200;

    @Parameter(names = {"-w"}, description = "Number of ParallelWrapper workers")
    private int numWorkers = 4;

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

        if (numWorkers < 2)
            throw new IllegalStateException("Number of workers should be < 2");

        // basically we're just going to fire PW here

        SleepingDataSetIterator iterator = new SleepingDataSetIterator(timeETL, 100);


        ParallelWrapper wrapper = new ParallelWrapper.Builder<>(Modeller.getConvolutionalModel())
                .workspaceMode(WorkspaceMode.SEPARATE)
                .averageUpdaters(false)
                .useLegacyAveraging(false)
                .averagingFrequency(5)
                .prefetchBuffer(8)
                .workers(numWorkers)
                .useMQ(false)
                .trainerFactory(new SleepingTrainerContext(timeFF, timeBP))
                .build();


        wrapper.fit(iterator);

        // and here we'll print out statistics
        log.info("Simulation finished");
    }

    public static void main(String[] args) throws Exception {
        new StandaloneSimulator().run(args);
    }
}
