# SUT plugins

ModelGauge uses [namespace plugins](../../docs/plugins.md) to separate the core libraries from the implementations of specific SUTs. That way you only have to install the dependencies you actually care about.

Any file put in this directory, or in any installed package with a namespace of `modelgauge.suts`, will be automatically loaded by the ModelGauge command line tool via `load_plugins()`.
