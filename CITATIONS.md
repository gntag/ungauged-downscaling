# Research Citations

Sources consulted during the design and improvement of the Delos statistical
climate downscaling pipeline.

---

## Terrain-based temperature downscaling

**TopoSCALE v1.0 — Downscaling gridded climate data in complex terrain**
Fiddes J. & Gruber S. (2014). *Geoscientific Model Development*, 7, 387–405.
https://gmd.copernicus.org/articles/7/387/2014/

Rationale for delta-elevation correction, sky view factor, and patch-based
pressure-level interpolation as core terrain predictors. Primary reference
for Propositions 1, 2, 7.

---

**TopoCLIM: Rapid topography-based downscaling of regional climate model output v1.1**
Filhol S. et al. (2022). *Geoscientific Model Development*, 15, 1753–1768.
https://gmd.copernicus.org/articles/15/1753/2022/

Extends TopoSCALE to CORDEX RCM output with quantile mapping. Shows that
without delta-elevation correction, quantile models consistently over-predict
temperatures at ridges and under-predict in basins. Reference for Propositions 1, 2.

---

**TopoPyScale: A Python Package for Hillslope Climate Downscaling**
Filhol S. et al. (2023). *Journal of Open Source Software*.
https://pypi.org/project/TopoPyScale/

Python implementation of TopoSCALE terrain parameters (SVF, slope, aspect,
horizon angles). Reference for Proposition 7 (SVF).

---

**Environmental Lapse Rate for High-Resolution Land Surface Downscaling: An Application to ERA5**
Dutra E. et al. (2020). *Earth and Space Science*, 7(5).
https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2019ea000984

Derives spatially and temporally variable ELR from ERA5 pressure-level columns.
Shows seasonal ELR range of −4 to −7 K/km in Europe; constant MALR is a poor
substitute for coastal and insular climates. Reference for Proposition 2.

---

**A Physically Based Algorithm for Downscaling Temperature in Complex Terrain**
Blandford T.R. et al. (2008). *Journal of Applied Meteorology and Climatology*, 57(8), 1907–1922.
https://journals.ametsoc.org/jamc/article/57/8/1907/68330/

Demonstrates that locally estimated monthly lapse rates reduce RMSE 15–40%
over fixed −6.5 K/km in complex terrain. Reference for Proposition 2.

---

**Topographic Visualization of Near-surface Temperatures for Improved Lapse Rate Estimation**
(2024). *arXiv*, 2406.11894.
https://arxiv.org/abs/2406.11894

Topographic features for improving lapse-rate estimation; discussion of
uncertainties from unresolved topography.

---

**Downscaling climate-model output in mountainous terrain using local topographic lapse rates**
(ResearchGate, 2008).
https://www.researchgate.net/publication/319891103

Application of local topographic lapse rates for hydrologic impact modelling.

---

## Coastal and sea-breeze effects

**Knowledge-based and Physiographic Weighting Functions for Climate Mapping**
Daly C. et al. (2003). *International Journal of Climatology*, 23, 1359–1381.
https://data.fs.usda.gov/research/pubs/iitf/ja_iitf_2003_daly001.pdf

Describes the PRISM coastal-influence trajectory model: cost-benefit path
analysis along prevailing winds, finding optimal path marine air takes from
the coast. Shows physiographic weighting (elevation, coastal proximity, aspect,
topographic position) captures the majority of temperature residuals that
elevation alone misses. Reference for Propositions 3, 5.

---

**The PRISM Approach to Mapping Precipitation and Temperature**
Daly C., Neilson R.P. & Phillips D.L. (1997). *Applied Climatology Conference Proceedings*.
https://prism.oregonstate.edu/pubs/link/1997_daly-etal_conf-appl-climat.pdf

Foundational PRISM paper; describes coast-distance, aspect, topographic-position
weighting functions. Reference for Propositions 3, 5, 6.

---

**North-western Mediterranean sea-breeze circulation in a regional climate system model**
Lebeaupin Brossier C. et al. (2018). *Climate Dynamics*, 51, 1517–1538.
https://link.springer.com/article/10.1007/s00382-017-3595-z

Documents sea-breeze signatures in regional climate models; confirms that sea-breeze
cooling requires directional fetch or explicit dynamics to be reproduced statistically.

---

**Effective Fetch and Relative Exposure Index Maps for the Laurentian Great Lakes**
Rao Y.R. et al. (2018). *PLOS ONE* / PMC.
https://pmc.ncbi.nlm.nih.gov/articles/PMC6298251/

Defines effective fetch as weighted sum of open-water distances in multiple compass
directions, weighted by cos(angle) relative to prevailing wind. Reference for
Proposition 3 (directional fetch implementation).

---

**Assessing present and future coastal moderation of extreme heat in the Eastern United States**
(2019). *Environmental Research Letters*.
https://iopscience.iop.org/article/10.1088/1748-9326/ab495d

Quantifies marine cooling as a function of coastal distance and fetch; confirms
straight-line coast distance misses directional shelter effects by 1–3 °C in summer.

---

**Statistical Downscaling of Gridded Wind Speed Data Using Local Topography**
Liston G.E. & Sturm M. (2017). *Journal of Hydrometeorology*, 18(2), 335–356.
https://journals.ametsoc.org/jhm/article/18/2/335/69671/

Wind downscaling with topographic exposure (Sx index) and directional shelter as
predictors. Reference for Propositions 4, 5.

---

## Topographic position and cold-air pooling

**Frequent and Strong Cold-Air Pooling Drives Temperate Forest Composition**
Holden Z.A. et al. (2011). *PMC / Global Change Biology*.
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10985370/

Shows TPI is the strongest predictor of cold-pool occurrence; negative TPI (valley)
strongly predicts temperature inversions; positive TPI (ridge) is warmer at night.
Reference for Proposition 6.

---

**Topography Influences Diurnal and Seasonal Microclimate in Hilly Coastal California**
(2024). *PMC*.
https://pmc.ncbi.nlm.nih.gov/articles/PMC10980203/

Elevation and hillslope position as primary drivers of daily temperature variation
near the coast; cold-air pooling conditions in convergent environments. Reference
for Proposition 6.

---

## Machine learning downscaling with terrain features

**Machine Learning Framework for High-Resolution Air Temperature Downscaling Using LiDAR-Derived Urban Morphological Features**
(2024). *arXiv*, 2409.02120.
https://arxiv.org/pdf/2409.02120

LightGBM with topographic indices (slope, aspect, elevation gradient, surface roughness)
achieves RMSE of 0.352 K for air temperature downscaling.

---

**GeoAI in Temperature Correction: Feature Importance with SHAP**
(2025). *IJGI*, 15(1).
https://doi.org/10.3390/ijgi15010031

SHAP analysis confirming elevation, slope, aspect, and TRI as dominant terrain
predictors in gradient-boosting temperature downscaling. Reference for Propositions 5, 6.

---

**Spatial Downscaling of Precipitation Data Based on XGBoost-MGWR**
(2024). *Land*, 13(4).
https://www.mdpi.com/2073-445X/13/4/448

XGBoost downscaling using TPI at multiple scales; demonstrates value of multi-scale
terrain position features for precipitation. Reference for Proposition 6.

---

## CERRA reanalysis validation

**Validating the Copernicus European Regional Reanalysis (CERRA) for Human-Biometeorological Applications**
(2023). *MDPI Atmosphere*, 26(1).
https://www.mdpi.com/2673-4931/26/1/111

CERRA validation at 35 Greek stations; cold-biased by up to 2 °C, with clear
elevation-dependent bias increase. Motivates delta-elevation correction (Proposition 1).

---

**CERRA, the Copernicus European Regional Reanalysis System**
Ridal M. et al. (2024). *Quarterly Journal of the Royal Meteorological Society*.
https://rmets.onlinelibrary.wiley.com/doi/10.1002/qj.4764

Official CERRA system description; confirms ~5 km native grid, orography storage,
and analysis/forecast stream architecture used by cerra_extract.py.

---

**Comparison of High-Resolution Climate Reanalysis Datasets for Hydro-Climatic Impact Studies**
(2025). *Hydrology and Earth System Sciences*, 29, 4153.
https://hess.copernicus.org/articles/29/4153/2025/

CERRA-Land identified as most reliable for surface temperature; bias grows with
elevation. Reference motivating Proposition 1.

---

**Pan-European High-Resolution Downscaling Using Deep Learning**
(2025). *JGR Atmospheres*.
https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2025JH000630

Deep-learning downscaling of CERRA to sub-kilometre resolution over Europe;
terrain predictors significantly improve upon plain CERRA interpolation.

---

## Longwave radiation and sky view factor

**Topographic Radiation Modeling and Spatial Scaling of Clear-Sky Land Surface Longwave Radiation over Rugged Terrain**
(ResearchGate).
https://www.researchgate.net/publication/284095517

SVF controls incoming longwave radiation and nocturnal cooling rates;
primary reference for Proposition 7.

---

**Modeling Surface Longwave Radiation over High-Relief Terrain**
(2019). *Remote Sensing of Environment*.
https://www.sciencedirect.com/science/article/abs/pii/S0034425719305760

Parameterization scheme for SVF-based longwave downwelling; shows sky view factor
dominates nocturnal Tmin depression in valleys. Reference for Proposition 7.

---

## Precipitation distribution correction (quantile mapping)

**Technical Note: Downscaling RCM precipitation to the station scale using statistical
transformations — a comparison of methods**
Gudmundsson L. et al. (2012). *Hydrology and Earth System Sciences*, 16, 3383–3390.
https://hess.copernicus.org/articles/16/3383/2012/

Empirical quantile mapping with wet-day frequency correction (the `fitQmapQUANT`
method). Basis for the wet-day threshold handling in `bias_correct.fit_eqm` and the
interpolation QM (`seasonal_precip_eqm`) used inside the final precipitation blend.

---

**Empirical-statistical downscaling and error correction of daily precipitation from
regional climate models**
Themeßl M. J., Gobiet A. & Leuprecht A. (2011). *International Journal of Climatology*,
31(10), 1530–1544. https://doi.org/10.1002/joc.2168

Per-day-of-year window quantile mapping; motivates the smooth ±45-day DOY windows in
`seasonal_precip_eqm` rather than hard seasonal bins.

---

**Bias correction of GCM precipitation by quantile mapping: How well do methods
preserve changes in quantiles and extremes?**
Cannon A. J., Sobie S. R. & Murdock T. Q. (2015). *Journal of Climate*, 28, 6938–6959.
https://doi.org/10.1175/JCLI-D-14-00754.1

Quantile delta mapping and the preservation of extremes under quantile mapping;
context for correcting the >20/>50 mm tail without distorting the marginal distribution.

---

**Stop using the RMSE for precipitation (target-metric mismatch)**
arXiv:2509.08369.

Argues the conditional mean, not the median, is the correct target when total-amount
preservation matters — rationale for training the Gamma conditional-mean `precip_mean`
rather than relying on the p50 median as the amount estimate.

---

## Precipitation and wind machine-learning downscaling

**Multi-quantile regression for extreme precipitation downscaling (Q-SRDRN)**
arXiv:2605.12762.

Extreme-event sample upweighting for the upper quantiles — basis for the
`EXTREME_SAMPLE_WEIGHT` upweighting of p90/p95/p99 wet-day training.

---

**Evaluating machine-learning models for wind-speed downscaling from ECMWF-IFS data**
Ericson et al. (2025). *Quarterly Journal of the Royal Meteorological Society*.
https://doi.org/10.1002/qj.5063

LightGBM wind downscaling; 10 m u/v components as primary directional predictors —
motivates adding `cerra_u10`/`cerra_v10` to the wind/gust feature sets.

---

**A decision-tree-based measure-correlate-predict approach for peak wind-gust
estimation (INTRIGUE)**
Wind Energy Science (2023), 8, 1533. https://wes.copernicus.org/articles/8/1533/2023/

Gradient-boosting gust modelling; the gust factor `gust/wind` encodes boundary-layer
state — basis for the derived `gust_factor_cerra` predictor.
