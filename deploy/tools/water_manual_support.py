"""Optional conservative abstention for weak, fragmented body colour evidence."""
def colour_support(colour, valid, reference, ref_valid, expected, spec):
    def weak_stations(ratios, mask):
        weak = mask & (ratios[:,:,expected] >= spec.colour_min)
        weak &= ratios[:,:,expected] > ratios[:,:,1-expected]
        return int(weak.any(axis=1).sum())

    current = weak_stations(colour,valid)
    baseline = weak_stations(reference[:,:,:2],ref_valid)
    # Weak red in a labelled body is evidence of poor recognition, not absence.
    ambiguous = max(current,baseline) >= spec.region_weak_stations
    return dict(current_weak_stations=current,reference_weak_stations=baseline,
                ambiguous=bool(ambiguous),minimum_stations=spec.region_weak_stations)
