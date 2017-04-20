package org.deeplearning4j.etl.tools;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.parallelism.MagicQueue;
import org.deeplearning4j.parallelism.ParallelWrapper;
import org.deeplearning4j.parallelism.factory.TrainerContext;
import org.deeplearning4j.parallelism.trainer.Trainer;

/**
 * Created by raver119 on 20.04.17.
 */
@Slf4j
public class SleepingTrainerContext implements TrainerContext {

    private long trainingTime;

    public SleepingTrainerContext(long trainingTime) {
        this.trainingTime = trainingTime;
    }

    /**
     * Initialize the context
     *
     * @param model
     * @param args  the arguments to initialize with (maybe null)
     */
    @Override
    public void init(Model model, Object... args) {

    }

    /**
     * Create a {@link Trainer}
     * based on the given parameters
     *
     * @param threadId      the thread id to use for this worker
     * @param model         the model to start the trainer with
     * @param rootDevice    the root device id
     * @param useMDS        whether to use the {@link MagicQueue}
     *                      or not
     * @param wrapper       the wrapper instance to use with this trainer (this refernece is needed
     *                      for coordination with the {@link ParallelWrapper} 's {@link IterationListener}
     * @param workspaceMode
     * @return the created training instance
     */
    @Override
    public Trainer create(int threadId, Model model, int rootDevice, boolean useMDS, ParallelWrapper wrapper, WorkspaceMode workspaceMode) {
        log.info("Creating new SleepingTraner: {}", threadId);
        return new SleepingTrainer(threadId, model, trainingTime);
    }
}
