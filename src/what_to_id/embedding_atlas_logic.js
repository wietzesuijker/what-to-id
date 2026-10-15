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
                             order = 'sample', hasReference = false, onlyBrushed = false, brushedIndices = []} = {}) {
    const query = term.trim().toLocaleLowerCase();
    const brush = new Set(brushedIndices);
    const result = records.map((r, i) => ({r, i})).filter(({r, i}) =>
      (!onlyBrushed || brush.has(i)) &&
      (!onlyInView || inBounds(r, bounds)) &&
      `${r.name} ${r.id}`.toLocaleLowerCase().includes(query));
    if (order === 'reference' && hasReference) {
      result.sort((a, b) => b.r.reference_distance - a.r.reference_distance || a.r.id - b.r.id);
    }
    return result;
  }
  function brushIndices(points, rectangle) {
    if (!rectangle || rectangle.length !== 4 || !rectangle.every(Number.isFinite)) return [];
    const [x0, y0, x1, y1] = rectangle;
    const left = Math.min(x0, x1), right = Math.max(x0, x1);
    const top = Math.min(y0, y1), bottom = Math.max(y0, y1);
    return points.flatMap(([x, y], i) => Number.isFinite(x) && Number.isFinite(y) &&
      x >= left && x <= right && y >= top && y <= bottom ? [i] : []);
  }
  function toggleIndex(indices, index) {
    const result = new Set(indices);
    if (result.has(index)) result.delete(index); else result.add(index);
    return [...result].sort((a, b) => a - b);
  }
  function modelHref(relative, base, group, id) {
    const target = new URL(relative, base);
    const state = new URLSearchParams();
    state.set('group', group);
    if (id !== null && id !== undefined) state.set('id', String(id));
    target.hash = state.toString();
    return target.href;
  }
  function mapHref(target, base, group, id) {
    const url = new URL(target, base);
    const state = new URLSearchParams({groups: group});
    if (id !== null && id !== undefined) state.set('record', String(id));
    url.hash = state.toString();
    return url.href;
  }
  function resolveSelection(records, requested) {
    const id = Number(requested);
    const valid = requested !== null && requested !== '' && Number.isSafeInteger(id) && id > 0;
    if (!valid) return {index: 0, missingRequestedId: null};
    const index = records.findIndex(record => record.id === id);
    return {index: index < 0 ? 0 : index, missingRequestedId: index < 0 ? id : null};
  }
  function compareNeighborhoods(left, right, queryId) {
    const a = left.records.find(r => r.id === queryId);
    const b = right.records.find(r => r.id === queryId);
    if (!a || !b) return {status: 'missing_query'};
    const leftIds = new Set(left.records.map(r => r.id));
    const rightIds = new Set(right.records.map(r => r.id));
    if (left.group !== right.group || leftIds.size !== left.records.length ||
        rightIds.size !== right.records.length || leftIds.size !== rightIds.size ||
        [...leftIds].some(id => !rightIds.has(id))) return {status: 'pool_mismatch'};
    if (leftIds.size < 2) return {status: 'insufficient_pool'};
    if (left.backbone === right.backbone) return {status: 'same_encoder'};
    function neighbors(payload, record) {
      if (!Array.isArray(record.neighbors)) return null;
      const k = Math.min(12, payload.records.length - 1);
      if (record.neighbors.length < k) return null;
      const result = [];
      for (const entry of record.neighbors.slice(0, k)) {
        const index = Array.isArray(entry) ? entry[0] : null;
        if (!Number.isInteger(index) || !payload.records[index]) return null;
        const id = payload.records[index].id;
        if (id === queryId || result.includes(id)) return null;
        result.push(id);
      }
      return result.length ? result : null;
    }
    const first = neighbors(left, a), second = neighbors(right, b);
    if (!first || !second) return {status: 'invalid_neighbors'};
    const other = new Set(second);
    return {status: 'comparable', leftCount: first.length, rightCount: second.length,
      sharedIds: first.filter(id => other.has(id)), cohortCount: leftIds.size,
      exhaustive: leftIds.size <= 13};
  }
  function requestGate() {
    let generation = 0;
    return {begin: () => ++generation, cancel: () => {generation++;},
      current: token => token === generation};
  }
  return {inBounds, matches, brushIndices, toggleIndex, modelHref, resolveSelection, mapHref,
    compareNeighborhoods, requestGate};
})();
if (typeof module !== 'undefined') module.exports = AtlasScope;
