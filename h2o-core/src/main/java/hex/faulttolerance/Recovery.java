package hex.faulttolerance;

import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;
import hex.Model;
import hex.grid.Grid;
import hex.grid.GridSearch;
import org.apache.log4j.Logger;
import water.*;
import water.api.GridSearchHandler;
import water.fvec.Frame;
import water.fvec.persist.FramePersist;
import water.fvec.persist.PersistUtils;
import water.util.FileUtils;
import water.util.IcedHashMap;

import java.io.IOException;
import java.net.URI;
import java.util.*;

public class Recovery<T extends Keyed> {
    
    private static final Logger LOG = Logger.getLogger(Recovery.class);

    public static final String REFERENCES_META_FILE_SUFFIX = "_references";
    public static final String RECOVERY_META_FILE = "recovery.json";
    
    public static final String INFO_CLASS = "class";
    public static final String INFO_RESULT_KEY = "resultKey";
    public static final String INFO_JOB_KEY = "jobKey";

    public static void autoRecover(String autoRecoveryDir) {
        if (autoRecoveryDir == null || autoRecoveryDir.length() == 0) {
            LOG.debug("Auto recovery dir not configured.");
        } else {
            LOG.info("Initializing auto recovery from " + autoRecoveryDir);
            H2O.submitTask(new H2O.H2OCountedCompleter(H2O.MIN_PRIORITY) {
                @Override
                public void compute2() {
                    new Recovery(autoRecoveryDir).autoRecover();
                    tryComplete();
                }
            });
        }
    }

    public enum ReferenceType {
        FRAME, KEYED
    }

    private final String storagePath;
    private final List<String> writtenFiles = new ArrayList<>(); 

    /**
     * @param storagePath directory to use as base for recovery snapshots
     */
    public Recovery(String storagePath) {
        this.storagePath = storagePath;
    }
    
    private String recoveryFile(String f) {
        return storagePath + "/" + f;
    }

    private String recoveryFile(Key key) {
        return recoveryFile(key.toString());
    }
    
    public String referencesMetaFile(Recoverable<T> r) {
        return recoveryFile(r.getKey().toString() + REFERENCES_META_FILE_SUFFIX);
    }
    
    public String recoveryMetaFile() {
        return recoveryFile(RECOVERY_META_FILE);
    }

    /**
     * Called when the training begins, so that initial state can be persisted
     * 
     * @param r a Recoverable to persist
     */
    public void onStart(final Recoverable<T> r, final Job job) {
        writtenFiles.addAll(r.exportBinary(storagePath, true));
        exportReferences(r);
        writeRecoveryInfo(r, job.getKey());
    }

    /**
     * Called by the Recoverable to notify of new model was trained and needs to persisted
     * 
     * @param r a Recoverable to update
     * @param modelKey key of the newly trained model
     */
    public void onModel(final Recoverable<T> r, Key<Model> modelKey) {
        try {
            String modelFile = recoveryFile(modelKey);
            modelKey.get().exportBinaryModel(modelFile, true);
            writtenFiles.add(modelFile);
            r.exportBinary(storagePath, false);
        } catch (IOException e) {
            // this should not happen since storagePath should be writable because
            // grid was already written to it
            throw new RuntimeException("Failed to store model for fault tolerance.", e);
        }
    }

    /**
     * Called by the recoverable that the training was finished successfully. This means that
     * recovery snapshots (persisted data) is no longer needed and can be deleted.
     */
    public void onDone(Recoverable<T> r) {
        final URI storageUri = FileUtils.getURI(storagePath);
        for (String path : writtenFiles) {
            URI pathUri = FileUtils.getURI(path);
            H2O.getPM().getPersistForURI(storageUri).delete(pathUri.toString());
        }
    }

    /**
     * Saves all of the keyed objects used by this Grid's params. Files are named by objects' keys.
     */
    public void exportReferences(final Recoverable<T> r) {
        final Set<Key<?>> keys = r.getDependentKeys();
        final IcedHashMap<String, String> referenceKeyTypeMap = new IcedHashMap<>();
        for (Key<?> k : keys) {
            persistObj(k.get(), referenceKeyTypeMap);
        }
        final URI referencesUri = FileUtils.getURI(referencesMetaFile(r));
        writtenFiles.add(referencesUri.toString());
        PersistUtils.write(referencesUri, ab -> ab.put(referenceKeyTypeMap));
    }
    
    public void writeRecoveryInfo(final Recoverable<T> r, Key<Job> jobKey) {
        Map<String, String> info = new HashMap<>();
        info.put(INFO_CLASS, r.getClass().getName());
        info.put(INFO_JOB_KEY, jobKey.toString());
        info.put(INFO_RESULT_KEY, r.getKey().toString());
        final URI infoUri = FileUtils.getURI(recoveryMetaFile());
        writtenFiles.add(infoUri.toString());
        PersistUtils.writeStream(infoUri, w -> w.write(new Gson().toJson(info)));
    }

    private void persistObj(
        final Keyed<?> o,
        Map<String, String> referenceKeyTypeMap
    ) {
        if (o instanceof Frame) {
            referenceKeyTypeMap.put(o._key.toString(), ReferenceType.FRAME.toString());
            String[] writtenFrameFiles = new FramePersist((Frame) o).saveToAndWait(storagePath, true);
            writtenFiles.addAll(Arrays.asList(writtenFrameFiles));
        } else if (o != null) {
            referenceKeyTypeMap.put(o._key.toString(), ReferenceType.KEYED.toString());
            String destFile = storagePath + "/" + o._key;
            URI dest = FileUtils.getURI(destFile);
            PersistUtils.write(dest, ab -> ab.putKey(o._key));
            writtenFiles.add(destFile);
        }
    }

    public void loadReferences(final Recoverable<T> r) {
        final URI referencesUri = FileUtils.getURI(storagePath + "/" + r.getKey() + REFERENCES_META_FILE_SUFFIX);
        Map<String, String> referencesMap = PersistUtils.read(referencesUri, AutoBuffer::get);
        final Futures fs = new Futures();
        referencesMap.forEach((key, type) -> {
            switch (ReferenceType.valueOf(type)) {
                case FRAME: 
                    FramePersist.loadFrom(Key.make(key), storagePath).get();
                    break;
                case KEYED:
                    PersistUtils.read(URI.create(storagePath + "/" + key), ab -> ab.getKey(Key.make(key), fs));
                    break;
                default:
                    throw new IllegalStateException("Unknown reference type " + type);
            }
        });
        fs.blockForPending();
    }
    
    public void autoRecover() {
        URI recoveryMetaUri = FileUtils.getURI(recoveryMetaFile());
        if (!PersistUtils.exists(recoveryMetaUri)) {
            return;
        }
        Map<String, String> recoveryInfo = PersistUtils.readStream(
            recoveryMetaUri, 
            r -> new Gson().fromJson(r, new TypeToken<Map<String, String>>(){}.getType())
        ); 
        String className = recoveryInfo.get(INFO_CLASS);
        Key<Job> jobKey = Key.make(recoveryInfo.get(INFO_JOB_KEY));
        Key<?> resultKey = Key.make(recoveryInfo.get(INFO_RESULT_KEY));
        if (Grid.class.getName().equals(className)) {
            Grid grid = Grid.importBinary(recoveryFile(resultKey), true);
            GridSearch.resumeGridSearch(
                jobKey, grid,
                new GridSearchHandler.DefaultModelParametersBuilderFactory(),
                (Recovery<Grid>) this
            );
        } else {
            LOG.error("Unable to recover object of class " + className);
        }
    }

}
