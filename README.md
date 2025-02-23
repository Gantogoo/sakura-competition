# Cherry Blossom Peak Bloom Prediction 2025

Entry repository for George Mason’s Department of Statistics cherry blossom peak bloom prediction competition.


## Getting started

Introduction to how to use.

## Competition rules

To enter the competition you must submit your predictions and the URL pointing to your repository via https://competition.statistics.gmu.edu.

**Entries must be submitted by the end of February 28, 2025 (anywhere on earth)**.
If it's February anywhere on earth, your submission will be considered.

The predictions are judged based on the sum of the absolute differences between your predicted peak bloom dates and the publicly posted peak bloom dates:

```
| predicted_bloom_date_kyoto_2025 - actual_bloom_date_kyoto_2025 | +
  | predicted_bloom_date_washingtondc_2025 - actual_bloom_date_washingtondc_2025 | +
  | predicted_bloom_date_liestal_2025 - actual_bloom_date_liestal_2025 | +
  | predicted_bloom_date_vancouver_2025 - actual_bloom_date_vancouver_2025 | +
  | predicted_bloom_date_newyorkcity_2025 - actual_bloom_date_newyorkcity_2025 |
```

The prediction intervals are evaluated based on how many out of the five intervals cover the actual bloom date.
In the case of ties, the sum of the squared lengths of the intervals is used to break the ties.

The true bloom dates for 2025 are taken to be the dates reported by the following agencies/institutions:

- **Kyoto (Japan):** a local news paper from Arashiyama (See http://atmenv.envi.osakafu-u.ac.jp/aono/kyophenotemp4),
- **Washington, D.C. (USA):** National Park Service,
- **Liestal-Weideli (Switzerland):** MeteoSwiss,
- **Vancouver, BC (Canada):** Vancouver Cherry Blossom Festival in collaboration with Douglas Justice, Associate Director, Curator of Collections, UBC Botanical Garden.
- **New York City, NY (USA):** The Washington Square Park Eco Projects in collaboration with the Nature Lab.

The full competition rules are available under https://competition.statistics.gmu.edu.

## License

![CC-BYNCSA-4](https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png)

Unless otherwise noted, the content in this repository is licensed under a [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](http://creativecommons.org/licenses/by-nc-sa/4.0/).

For the data sets in the _data/_ folder, please see [_data/README.md_](data/README.md) for the applicable copyrights and licenses.
