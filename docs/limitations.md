# Limitations

LaFGS requires posed mapping images, sufficient cross-view overlap for stable
tracks, and a Gaussian prior whose raster support is broadly aligned with the
mapping coordinate frame. It reduces propagation of primitive-level errors but
is not independent of prior geometry or visibility.

Repeated structures can still create false global top-1 assignments. The
release intentionally does not hide this with dense refinement, learned pose
sampling, test-time rendering, or multiple custom solvers. Tail errors should
therefore be reported alongside median error.

Map reconstruction is offline and currently GPU intensive. Feed-forward
Gaussian priors reduce scaffold construction cost, but LaFGS evidence building
and descriptor reconstruction are not real-time map-building processes.
