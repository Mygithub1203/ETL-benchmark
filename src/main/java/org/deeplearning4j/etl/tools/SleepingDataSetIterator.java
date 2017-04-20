package org.deeplearning4j.etl.tools;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;

import java.util.List;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.locks.LockSupport;

/**
 * @author raver119@gmail.com
 */
@Slf4j
public class SleepingDataSetIterator implements DataSetIterator {
    private long sleep;
    private AtomicLong counter = new AtomicLong(0);
    private long limit;

    private final long startingTime;

    public SleepingDataSetIterator(long sleepTimeMillis, long numberOfExamples) {
        this.sleep = sleepTimeMillis;
        this.limit = numberOfExamples;

        this.startingTime = System.currentTimeMillis();
    }

    @Override
    public DataSet next(int i) {
        throw new UnsupportedOperationException();
    }

    @Override
    public int totalExamples() {
        return 0;
    }

    @Override
    public int inputColumns() {
        return 0;
    }

    @Override
    public int totalOutcomes() {
        return 0;
    }

    @Override
    public boolean resetSupported() {
        return true;
    }

    @Override
    public boolean asyncSupported() {
        return true;
    }

    @Override
    public void reset() {
        this.counter.set(0);
    }

    @Override
    public int batch() {
        return 0;
    }

    @Override
    public int cursor() {
        return 0;
    }

    @Override
    public int numExamples() {
        return 0;
    }

    @Override
    public void setPreProcessor(DataSetPreProcessor dataSetPreProcessor) {

    }


    public void printOutThroughput() {
        long finalTime = System.currentTimeMillis();
        long delta = finalTime - startingTime;
        log.info("{} datasets were processed; {} datasets/second", counter.get(), counter.get() / (delta / 1000));
    }

    @Override
    public DataSetPreProcessor getPreProcessor() {
        return null;
    }

    @Override
    public List<String> getLabels() {
        return null;
    }

    /**
     * Returns {@code true} if the iteration has more elements.
     * (In other words, returns {@code true} if {@link #next} would
     * return an element rather than throwing an exception.)
     *
     * @return {@code true} if the iteration has more elements
     */
    @Override
    public boolean hasNext() {
        return counter.get() < limit;
    }

    /**
     * Returns the next element in the iteration.
     *
     * @return the next element in the iteration
     */
    @Override
    public DataSet next() {
        counter.incrementAndGet();

        INDArray features = Nd4j.create(8, 784);
        INDArray labels = Nd4j.create(8, 10);
        DataSet ds = new DataSet(features, labels);

        LockSupport.parkNanos(sleep * 1000000L);

        if (counter.get() % 1000 == 0)
            printOutThroughput();

        return ds;
    }

    /**
     * Removes from the underlying collection the last element returned
     * by this iterator (optional operation).  This method can be called
     * only once per call to {@link #next}.  The behavior of an iterator
     * is unspecified if the underlying collection is modified while the
     * iteration is in progress in any way other than by calling this
     * method.
     *
     * @throws UnsupportedOperationException if the {@code remove}
     *                                       operation is not supported by this iterator
     * @throws IllegalStateException         if the {@code next} method has not
     *                                       yet been called, or the {@code remove} method has already
     *                                       been called after the last call to the {@code next}
     *                                       method
     * @implSpec The default implementation throws an instance of
     * {@link UnsupportedOperationException} and performs no other action.
     */
    @Override
    public void remove() {

    }
}
