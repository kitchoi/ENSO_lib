.. ENSO Diagnostic Tool documentation Home page

ENSO Diagnostic Tool's documentation
================================================

This is a python library for diagnosing El Nino-Southern Oscillation (ENSO) events and analysing their characteristics.  

Here are a few of the functions of this library:

* Identify ENSO events using an SST anomaly index
* Compute the composite of a physical variable for an El Nino or a La Nina event
* Compute the ENSO asymmetries in terms of amplitude, duration and transition preferences (`Choi et al. 2013 <http://dx.doi.org/10.1175/JCLI-D-13-00045.1>`_)
* Simulate ENSO SST anomalies under a conceptual framework with or without nonlinear air-sea coupling (`Choi et al. 2013 <http://dx.doi.org/10.1175/JCLI-D-13-00045.1>`_)
* Diagnostics for the maximum zonal wind response to ENSO taking into account varying spatial structures
* Statistical analysis for the spatial structure of ITCZ and SPCZ rainbands

Some of the functions are applicable for many other geophysical quantities outside of ENSO research.  Some are more focused on atmospheric responses relevant to ENSO.  Applications of these functions are documented in `Choi et al (2013) <http://dx.doi.org/10.1175/JCLI-D-13-00045.1>`_ and `Choi et al (2015) <http://dx.doi.org/10.1175/JCLI-D-15-0211.1>`_.  A few of the functions require the `GeoDAT library </geodat_doc>`_.


Contents:
=============
.. toctree::
   :maxdepth: 2

   diag
   maxu_r
   ITCZ
   Oscillator



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

