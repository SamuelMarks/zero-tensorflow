Extracting target APIs from /Users/samuel/repos/zero-tensorflow/src...
Scoring compliance...

--- Compliance Report ---
Overall Compliance: 101.51%

Breakdown by Module:
  - tf.data: 100.0% (10/10)
  - tf.keras.initializers: 100.0% (13/13)
  - tf.keras.layers: 100.0% (153/153)
  - tf.keras.losses: 100.0% (24/24)
  - tf.keras.metrics: 100.0% (46/46)
  - tf.keras.optimizers: 100.0% (15/15)
  - tf.keras.optimizers.schedules: 100.0% (1/1)
  - tf.nn: 100.0% (7/7)

Mismatched APIs (35):
  ~ tf.keras.initializers.GlorotNormal
  ~ tf.keras.initializers.GlorotUniform
  ~ tf.keras.initializers.HeNormal
  ~ tf.keras.initializers.HeUniform
  ~ tf.keras.initializers.LecunNormal
  ~ tf.keras.initializers.LecunUniform
  ~ tf.keras.initializers.Orthogonal
  ~ tf.keras.initializers.OrthogonalInitializer
  ~ tf.keras.initializers.RandomNormal
  ~ tf.keras.initializers.RandomUniform
  ~ tf.keras.initializers.TruncatedNormal
  ~ tf.keras.initializers.VarianceScaling
  ~ tf.keras.initializers.glorot_normal
  ~ tf.keras.initializers.glorot_uniform
  ~ tf.keras.initializers.he_normal
  ~ tf.keras.initializers.he_uniform
  ~ tf.keras.initializers.lecun_normal
  ~ tf.keras.initializers.lecun_uniform
  ~ tf.keras.initializers.orthogonal
  ~ tf.keras.initializers.random_normal
  ~ ... and 15 more
