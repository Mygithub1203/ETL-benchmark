package org.deeplearning4j.etl;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.ParameterException;
import lombok.extern.slf4j.Slf4j;

/**
 * @author raver119@gmail.com
 */
@Slf4j
public class Simulator {

    @Parameter(names = {"-d"}, description = "Time required to produce one DataSet")
    private int datasetTimeMillis = 10;

    @Parameter(names = {"-w"}, description = "Number of ParallelWrapper workers")
    private int numWorkers = 4;

    @Parameter(names = {"-t"}, description = "Training time of single ParallelWrapper worker")
    private int trainingTimeMillis = 50;


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


    }

    public static void main(String[] args) throws Exception {
        new Simulator().run(args);
    }
}
