package org.deeplearning4j.etl.tools;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.parallelism.trainer.Trainer;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.MultiDataSet;

import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.locks.LockSupport;

/**
 * @author raver119@gmail.com
 */
@Slf4j
public class SleepingTrainer extends Thread implements Trainer {

    private long trainingTime;
    private final AtomicBoolean locker = new AtomicBoolean(false);
    private final AtomicBoolean shouldWork = new AtomicBoolean(true);
    private final Model model;
    private final int threadId;

    public SleepingTrainer(int threadId, Model model, long trainingTime) {
        this.trainingTime = trainingTime;
        this.model = model;
        this.threadId = threadId;
    }

    /**
     * Train on a {@link MultiDataSet}
     *
     * @param dataSet the data set to train on
     */
    @Override
    public void feedMultiDataSet(MultiDataSet dataSet, long time) {
        locker.set(true);
    }

    /**
     * Train on a {@link DataSet}
     *
     * @param dataSet the data set to train on
     */
    @Override
    public void feedDataSet(DataSet dataSet, long time) {
        locker.set(true);
    }

    /**
     * THe current model for the trainer
     *
     * @return the current  {@link Model}
     * for the worker
     */
    @Override
    public Model getModel() {
        return model;
    }

    /**
     * Update the current {@link Model}
     * for the worker
     *
     * @param model the new model for this worker
     */
    @Override
    public void updateModel(Model model) {
        // no-op
    }

    @Override
    public boolean isRunning() {
        return shouldWork.get();
    }

    /**
     * Shutdown this worker
     */
    @Override
    public void shutdown() {
        shouldWork.set(false);
    }

    /**
     * Block the main thread
     * till the trainer is up and running.
     */
    @Override
    public void waitTillRunning() {
        while (locker.get())
            LockSupport.parkNanos(1000L);
    }

    /**
     * Set the {@link Thread.UncaughtExceptionHandler}
     * for this {@link Trainer}
     *
     * @param handler the handler for uncaught errors
     */
    @Override
    public void setUncaughtExceptionHandler(Thread.UncaughtExceptionHandler handler) {

    }

    /**
     * Start this trainer
     */
    @Override
    public void start() {
        super.start();
        shouldWork.set(true);
    }

    /**
     * When an object implementing interface <code>Runnable</code> is used
     * to create a thread, starting the thread causes the object's
     * <code>run</code> method to be called in that separately executing
     * thread.
     * <p>
     * The general contract of the method <code>run</code> is that it may
     * take any action whatsoever.
     *
     * @see Thread#run()
     */
    @Override
    public void run() {
        while (shouldWork.get()) {
            if (locker.get()) {
                LockSupport.parkNanos(trainingTime * 1000000L);
                locker.set(false);
            } else {
                LockSupport.parkNanos(1000L);
            }
        }
    }
}
