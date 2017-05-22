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
 * @author raver119@gmail.com
 */
@Slf4j
public class SleepingTrainerContext implements TrainerContext {

    private long trainingTime;
    private long ffTime;
    private long bpTime;

    public SleepingTrainerContext(long ffTime, long bpTime) {
        this.trainingTime = ffTime + bpTime;
        this.bpTime = bpTime;
        this.ffTime = ffTime;
    }

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
    public Trainer create(int threadId, Model model, int rootDevice, boolean useMDS, ParallelWrapper wrapper, WorkspaceMode workspaceMode, int averagingFrequency) {
        SleepingTrainer trainer = new SleepingTrainer(threadId, model, ffTime, bpTime);
        trainer.setDaemon(true);
        return trainer;
    }
}
