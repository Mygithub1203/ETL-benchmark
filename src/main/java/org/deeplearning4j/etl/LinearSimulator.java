package org.deeplearning4j.etl;

import com.beust.jcommander.Parameter;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.factory.Nd4j;

/**
 * @author raver119@gmail.com
 */
@Slf4j
public class LinearSimulator {
    @Parameter(names = {"-d"}, description = "Time required to produce one DataSet")
    private int datasetTimeMillis = 25;


    private int[] workers = new int[]{1, 2, 4, 6, 8};

    @Parameter(names = {"-t"}, description = "Training time of single ParallelWrapper worker")
    private int trainingTimeMillis = 100;

    @Parameter(names = {"-e"}, description = "Total number of examples to roll through")
    private int totalExamples = 1000;

    public void run(String[] args) throws Exception {
        Nd4j.create(1);

        log.info("Linear scaling simulation: {} ms per dataset, {} ms training time, {} total examples: -----------------------------", datasetTimeMillis, trainingTimeMillis, totalExamples);

        for (int w: workers) {
            new Simulator().run(new String[]{
                    "-d", "" + datasetTimeMillis,
                    "-t", "" + trainingTimeMillis,
                    "-w", "" + w,
                    "-e", "" + totalExamples
            });
        }
    }

    public static void main(String[] args) throws Exception {
        new LinearSimulator().run(args);
    }
}
