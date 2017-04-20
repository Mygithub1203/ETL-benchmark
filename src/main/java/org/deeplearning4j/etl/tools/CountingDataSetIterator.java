package org.deeplearning4j.etl.tools;

import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.atomic.AtomicLong;

/**
 * @author raver119@gmail.com
 */
@Slf4j
public class CountingDataSetIterator implements DataSetIterator {

    private long startingTime;
    private final AtomicLong counter = new AtomicLong(0);
    private final DataSetIterator iterator;
    private final List<Long> nanos = new ArrayList<>();

    public CountingDataSetIterator(@NonNull DataSetIterator iterator) {
        this.iterator = iterator;
        this.startingTime = System.currentTimeMillis();
    }


    @Override
    public DataSet next(int num) {
        throw new UnsupportedOperationException();
    }

    @Override
    public int totalExamples() {
        return iterator.totalExamples();
    }

    @Override
    public int inputColumns() {
        return iterator.inputColumns();
    }

    @Override
    public int totalOutcomes() {
        return iterator.totalOutcomes();
    }

    @Override
    public boolean resetSupported() {
        return iterator.resetSupported();
    }

    @Override
    public boolean asyncSupported() {
        return iterator.asyncSupported();
    }

    @Override
    public void reset() {
        startingTime = System.currentTimeMillis();
        nanos.clear();
        iterator.reset();
    }

    @Override
    public int batch() {
        return iterator.batch();
    }

    @Override
    public int cursor() {
        return iterator.cursor();
    }

    @Override
    public int numExamples() {
        return iterator.numExamples();
    }

    @Override
    public void setPreProcessor(DataSetPreProcessor preProcessor) {
        iterator.setPreProcessor(preProcessor);
    }

    @Override
    public DataSetPreProcessor getPreProcessor() {
        return iterator.getPreProcessor();
    }

    @Override
    public List<String> getLabels() {
        return iterator.getLabels();
    }

    @Override
    public boolean hasNext() {
        return iterator.hasNext();
    }

    @Override
    public DataSet next() {

        counter.incrementAndGet();
        long time1 = System.currentTimeMillis();
        DataSet ds = iterator.next();
        long time2 = System.currentTimeMillis();

        nanos.add(time2 - time1);

        return ds;
    }

    @Override
    public void remove() {
        // no-op
    }

    public long getThroughput() {
        long finalTime = System.currentTimeMillis();
        long delta = finalTime - startingTime;
        return counter.get() / (delta / 1000);
    }

    public void printOutThroughput() {
        Collections.sort(nanos);
        int pos = (int) (nanos.size() * 0.85);
        log.info("{} datasets were processed; {} datasets/second; p50: {} ms; p85: {} ms;", counter.get(), getThroughput(), nanos.get(nanos.size() / 2), nanos.get(pos));
    }
}
