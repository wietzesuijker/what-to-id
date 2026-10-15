/* Public-coordinate filtering, shared by the atlas and its Node regression tests. */
const AtlasScope = (() => {
  function inBounds(record, bounds) {
    if (!bounds || bounds.length !== 4 || !bounds.every(Number.isFinite)) return false;
    const [west, south, east, north] = bounds;
    if (!Number.isFinite(record.lon) || !Number.isFinite(record.lat) || south > north) return false;
    if (record.lat < south || record.lat > north) return false;
    let width = east - west;
    if (width >= 360) return true;
    if (width < 0) width = ((width % 360) + 360) % 360;
    const delta = (((record.lon - west) % 360) + 360) % 360;
    return delta <= width + 1e-9;
  }
  function matches(records, {term = '', bounds = null, onlyInView = false,
                             order = 'sample', hasReference = false} = {}) {
    const query = term.trim().toLocaleLowerCase();
    const result = records.map((r, i) => ({r, i})).filter(({r}) =>
      (!onlyInView || inBounds(r, bounds)) &&
      `${r.name} ${r.id}`.toLocaleLowerCase().includes(query));
    if (order === 'reference' && hasReference) {
      result.sort((a, b) => b.r.reference_distance - a.r.reference_distance || a.r.id - b.r.id);
    }
    return result;
  }
  return {inBounds, matches};
})();
if (typeof module !== 'undefined') module.exports = AtlasScope;
