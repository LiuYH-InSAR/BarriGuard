# Fixed lamp heatmap checks

One unique polygon or circle per lamp, using the original image dimensions.
Save under app/roi_configs/lamps/1749.json or 1750.json.

    lamp_checks:
      enabled: true
      polygons: roi_configs/lamps/1749.json
      heatmap_ratio: 0.3
      lit_hours: 3

Hourly overlap uses the existing heatmap's filtered support:
keep > 0, duty >= RENDER_DUTY_LO, dim > 0, gate > 0, outside excluded pixels.
The existing heatmap algorithm, colors and render weights are unchanged.
Coverage = supported heatmap pixels inside the lamp / whole lamp polygon area.
Strictly more than 30% means lit for that hour. It is an area ratio, not
an intensity threshold. Sampling coverage, unmasked area and candidate overflow
are diagnostics only. Earlier min_coverage/min_pixels/min_flash_ratio/missing_hours
fields remain accepted for compatibility but no longer affect this decision.

Morning: at least three distinct hours lit = green; otherwise red.
Missing/silent hours do not count as lit; their hourly state remains unknown/gray.
Repeated results within one hour replace its existing record.
Red means fewer than three detected lit hours, not a confirmed electrical fault.
Old fixed-lamp history is reset on upgrade so the two rules are not mixed.

Hourly images retain heatmaps and lamp circles. Morning images contain only
background and lamp circles. No lamp names. Full per-hour overlap ratios,
counts and diagnostics are saved in fixed_lamps/fixed_lamp_hours JSON fields.
Other recording, retention and callback behavior is unchanged.
