/**
 * Node class for the document block tree.
 */
class BlockNode {
  constructor(x0, y0, x1, y1, parent = null) {
    this.x0 = Math.floor(x0);
    this.y0 = Math.floor(y0);
    this.x1 = Math.floor(x1);
    this.y1 = Math.floor(y1);
    this.parent = parent;
    this.children = [];
    this.lineIdx = [];
    this.numLines = 0;
    this.numVerticalLines = 0;
  }

  getCoords() {
    return [this.x0, this.y0, this.x1, this.y1];
  }

  append(child) {
    this.children.push(child);
  }

  isXSplit() {
    if (this.children.length === 0) return false;
    const [_, y0, __, y1] = this.getCoords();
    for (const child of this.children) {
      const [__, cy0, ___, cy1] = child.getCoords();
      if (y0 !== cy0 || y1 !== cy1) return false;
    }
    return true;
  }

  isVertical() {
    return this.numLines < this.numVerticalLines * 2;
  }
}

/**
 * Finds the minimum span in the histogram array.
 * Replaces the NumPy approach using basic loops.
 */
function calcMinSpan(hist) {
  if (hist.length <= 1) return [0, hist.length, hist[0] || 0];

  let minVal = Infinity;
  let maxVal = -Infinity;

  for (let i = 0; i < hist.length; i++) {
    if (hist[i] < minVal) minVal = hist[i];
    if (hist[i] > maxVal) maxVal = hist[i];
  }

  let startIdxs = [];
  let endIdxs = [];
  let inSpan = false;

  for (let i = 0; i < hist.length; i++) {
    let isMin = hist[i] === minVal;
    if (isMin && !inSpan) {
      startIdxs.push(i);
      inSpan = true;
    } else if (!isMin && inSpan) {
      endIdxs.push(i);
      inSpan = false;
    }
  }
  if (inSpan) endIdxs.push(hist.length);

  let maxLen = -1;
  let bestIdx = 0;
  for (let i = 0; i < startIdxs.length; i++) {
    let len = endIdxs[i] - startIdxs[i];
    if (len > maxLen) {
      maxLen = len;
      bestIdx = i;
    }
  }

  let score = maxVal > 0 ? -minVal / maxVal : 0;
  return [startIdxs[bestIdx] || 0, endIdxs[bestIdx] || hist.length, score];
}

function calcHist(table, width, x0, y0, x1, y1) {
  let xHist = new Float64Array(x1 - x0);
  let yHist = new Float64Array(y1 - y0);
  for (let y = y0; y < y1; y++) {
    for (let x = x0; x < x1; x++) {
      let val = table[y * width + x];
      xHist[x - x0] += val;
      yHist[y - y0] += val;
    }
  }
  return [xHist, yHist];
}

function split(parent, table, width, x0, y0, x1, y1) {
  x0 = x0 ?? parent.x0;
  y0 = y0 ?? parent.y0;
  x1 = x1 ?? parent.x1;
  y1 = y1 ?? parent.y1;

  if (!(x0 < x1 && y0 < y1)) return;
  if (
    x0 === parent.x0 &&
    y0 === parent.y0 &&
    x1 === parent.x1 &&
    y1 === parent.y1
  )
    return;

  let child = new BlockNode(x0, y0, x1, y1, parent);
  parent.append(child);
  blockXyCut(table, width, child);
}

function splitX(parent, table, width, val, x0, x1) {
  split(parent, table, width, undefined, undefined, x0, undefined);
  split(parent, table, width, x0, undefined, x1, undefined);
  split(parent, table, width, x1, undefined, undefined, undefined);
}

function splitY(parent, table, width, val, y0, y1) {
  split(parent, table, width, undefined, undefined, undefined, y0);
  split(parent, table, width, undefined, y0, undefined, y1);
  split(parent, table, width, undefined, y1, undefined, undefined);
}

function blockXyCut(table, width, meNode) {
  let [x0, y0, x1, y1] = meNode.getCoords();
  let [xHist, yHist] = calcHist(table, width, x0, y0, x1, y1);

  let [xBeg, xEnd, xVal] = calcMinSpan(xHist);
  let [yBeg, yEnd, yVal] = calcMinSpan(yHist);

  xBeg += x0;
  xEnd += x0;
  yBeg += y0;
  yEnd += y0;

  if (x0 === xBeg && x1 === xEnd && y0 === yBeg && y1 === yEnd) return;

  if (yVal < xVal) {
    splitX(meNode, table, width, xVal, xBeg, xEnd);
  } else if (xVal < yVal) {
    splitY(meNode, table, width, yVal, yBeg, yEnd);
  } else if (xEnd - xBeg < yEnd - yBeg) {
    splitY(meNode, table, width, yVal, yBeg, yEnd);
  } else {
    splitX(meNode, table, width, xVal, xBeg, xEnd);
  }
}

function getOptimalGrid(bboxes) {
  return 100 * Math.sqrt(bboxes.length);
}

function normalizeBboxes(bboxes, grid, scale = 1.0, tolerance = 0.25) {
  // Deep copy coordinates to avoid mutating original
  let norm = bboxes.map((b) => [...b]);

  // Make width and height non-negative
  for (let i = 0; i < norm.length; i++) {
    if (norm[i][0] > norm[i][2]) norm[i][2] = norm[i][0];
    if (norm[i][1] > norm[i][3]) norm[i][3] = norm[i][1];
  }

  // Dilation/Erosion
  if (scale !== 1.0) {
    let w = norm.map((b) => b[2] - b[0]);
    let h = norm.map((b) => b[3] - b[1]);

    let minDims = w.map((cw, i) => Math.min(cw, h[i]));
    let sortedDims = [...minDims].sort((a, b) => a - b);
    let m = sortedDims[Math.floor(sortedDims.length / 2)] || 0;

    let lower = m * (1.0 - tolerance);
    let upper = m * (1.0 + tolerance);

    for (let i = 0; i < norm.length; i++) {
      let cw = w[i],
        ch = h[i];
      if (cw < ch && cw >= lower && cw < upper) {
        let diff = Math.floor(((scale - 1.0) * cw) / 2);
        norm[i][0] -= diff;
        norm[i][2] += diff;
      } else if (ch < cw && ch >= lower && ch < upper) {
        let diff = Math.floor(((scale - 1.0) * ch) / 2);
        norm[i][1] -= diff;
        norm[i][3] += diff;
      }
    }
  }

  // Coarse-grain into grid space
  let xMin = Math.min(...norm.map((b) => b[0]));
  let yMin = Math.min(...norm.map((b) => b[1]));
  let xMax = Math.max(...norm.map((b) => b[2]));
  let yMax = Math.max(...norm.map((b) => b[3]));

  let wPage = xMax - xMin || 1;
  let hPage = yMax - yMin || 1;

  let xGrid = wPage < hPage ? grid : grid * (wPage / hPage);
  let yGrid = hPage < wPage ? grid : grid * (hPage / wPage);

  for (let i = 0; i < norm.length; i++) {
    norm[i][0] = Math.max(0, Math.floor(((norm[i][0] - xMin) * xGrid) / wPage));
    norm[i][1] = Math.max(0, Math.floor(((norm[i][1] - yMin) * yGrid) / hPage));
    norm[i][2] = Math.max(0, Math.floor(((norm[i][2] - xMin) * xGrid) / wPage));
    norm[i][3] = Math.max(0, Math.floor(((norm[i][3] - yMin) * yGrid) / hPage));
  }

  return norm;
}

function makeMeshTable(bboxes) {
  let xMax = Math.max(...bboxes.map((b) => b[2]), 0);
  let yMax = Math.max(...bboxes.map((b) => b[3]), 0);
  let width = xMax + 1;
  let height = yMax + 1;

  // Using a 1D array for 2D matrix (faster in JS)
  let table = new Uint8Array(width * height);

  for (let b of bboxes) {
    let [x0, y0, x1, y1] = b;
    for (let y = y0; y < y1; y++) {
      for (let x = x0; x < x1; x++) {
        table[y * width + x] = 1;
      }
    }
  }
  return { table, width, height };
}

function calcIou(box, boxes) {
  return boxes.map((b) => {
    let x0 = Math.max(box[0], b[0]);
    let y0 = Math.max(box[1], b[1]);
    let x1 = Math.min(box[2], b[2]);
    let y1 = Math.min(box[3], b[3]);

    let interArea = Math.max(0, x1 - x0 + 1) * Math.max(0, y1 - y0 + 1);

    // Note: Mathematical fix vs original python bug where `boxes[:, 0]` was used inside `box_area`
    let boxArea =
      Math.max(0, box[2] - box[0] + 1) * Math.max(0, box[3] - box[1] + 1);
    let bArea = Math.max(0, b[2] - b[0] + 1) * Math.max(0, b[3] - b[1] + 1);

    let denom = boxArea + bArea - interArea;
    return denom > 0 ? interArea / denom : 0;
  });
}

function getBlockNodeBboxes(root) {
  let bboxes = [];
  let routers = [];
  function collect(node, router) {
    if (node.children.length === 0) {
      bboxes.push(node.getCoords());
      routers.push(router);
    }
    for (let i = 0; i < node.children.length; i++) {
      collect(node.children[i], [...router, i]);
    }
  }
  collect(root, []);
  return { routers, bboxes };
}

function routeTree(root, router) {
  let node = root;
  for (let i of router) node = node.children[i];
  return node;
}

function assignBboxToNode(root, bboxes) {
  let { routers, bboxes: leafBboxes } = getBlockNodeBboxes(root);
  for (let i = 0; i < bboxes.length; i++) {
    let ious = calcIou(bboxes[i], leafBboxes);
    let maxIou = -1;
    let maxIdx = 0;
    for (let j = 0; j < ious.length; j++) {
      if (ious[j] > maxIou) {
        maxIou = ious[j];
        maxIdx = j;
      }
    }
    routeTree(root, routers[maxIdx]).lineIdx.push(i);
  }
}

function sortNodes(node, bboxes) {
  if (node.lineIdx.length > 0) {
    let numVertical = 0;
    for (let i of node.lineIdx) {
      let w = bboxes[i][2] - bboxes[i][0];
      let h = bboxes[i][3] - bboxes[i][1];
      if (w < h) numVertical++;
    }
    node.numLines = node.lineIdx.length;
    node.numVerticalLines = numVertical;

    if (node.numLines > 1) {
      // Replicates numpy lexsort
      node.lineIdx.sort((a, b) => {
        let bA = bboxes[a],
          bB = bboxes[b];
        if (node.isVertical()) {
          // Right-to-Left, Top-to-Bottom
          if (bA[0] !== bB[0]) return bB[0] - bA[0]; // Descending X
          return bA[1] - bB[1]; // Ascending Y
        } else {
          // Top-to-Bottom, Left-to-Right
          if (bA[1] !== bB[1]) return bA[1] - bB[1]; // Ascending Y
          return bA[0] - bB[0]; // Ascending X
        }
      });
    }
  } else {
    for (let child of node.children) {
      let [num, vNum] = sortNodes(child, bboxes);
      node.numLines += num;
      node.numVerticalLines += vNum;
    }
    if (node.isXSplit() && node.isVertical()) {
      node.children.reverse(); // Standardize Right-To-Left reading order
    }
  }
  return [node.numLines, node.numVerticalLines];
}

function getRanking(node, ranks, rank = 0) {
  for (let i of node.lineIdx) {
    ranks[i] = rank++;
  }
  for (let child of node.children) {
    rank = getRanking(child, ranks, rank);
  }
  return rank;
}

/**
 * Core algorithm orchestrator (equivalent to solve() from the python code)
 */
export function solveXYCut(bboxesCoords, grid = null, scale = 1.0) {
  if (bboxesCoords.length === 0) return [];

  let gridVal = grid || getOptimalGrid(bboxesCoords);
  let normalized = normalizeBboxes(bboxesCoords, gridVal, scale);
  let { table, width, height } = makeMeshTable(normalized);

  let root = new BlockNode(0, 0, width, height, null);
  blockXyCut(table, width, root);
  assignBboxToNode(root, normalized);
  sortNodes(root, normalized);

  let ranks = new Array(bboxesCoords.length).fill(-1);
  getRanking(root, ranks, 0);

  return ranks;
}

/**
 * Main JSON entry point. (Equivalent to eval_xml)
 * @param {Object} pageJson - Requires an object with a `.lines` array containing objects with x, y, w, h
 * @param {Object} [options] - `{ scale: number, sortObjects: boolean }`
 */
export function evalJson(pageJson, options = {}) {
  let { scale = 1.0, sortObjects = true } = options;

  if (!pageJson.lines || pageJson.lines.length === 0) return pageJson;

  // Convert to target array structure [x, y, x+w, y+h]
  let bboxesCoords = pageJson.lines.map((line) => [
    line.x,
    line.y,
    line.x + line.w,
    line.y + line.h,
  ]);

  // Apply the math engine
  let ranks = solveXYCut(bboxesCoords, null, scale);

  // Hydrate the JSON object with the generated orders
  for (let i = 0; i < pageJson.lines.length; i++) {
    pageJson.lines[i].order = ranks[i];
  }

  if (sortObjects) {
    pageJson.lines.sort((a, b) => a.order - b.order);
  }

  return pageJson;
}
