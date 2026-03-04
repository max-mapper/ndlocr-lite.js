import { solveXYCut } from "./xyCut.js";

/**
 * NDLOCR-Lite Browser Port
 */

// # Classes
// names:

var classes = {
  0: "text_block",
  1: "line_main",
  2: "line_caption",
  3: "line_ad",
  4: "line_note",
  5: "line_note_tochu",
  6: "block_fig",
  7: "block_ad",
  8: "block_pillar",
  9: "block_folio",
  10: "block_rubi",
  11: "block_chart",
  12: "block_eqn",
  13: "block_cfm",
  14: "block_eng",
  15: "block_table",
  16: "line_title",
};

export class NDLOCR {
  constructor() {
    this.deimSession = null;
    this.parseq30Session = null;
    this.parseq50Session = null;
    this.parseq100Session = null;
    this.charlist = [];

    // Thresholds matching ocr.py
    this.detConfThreshold = 0.25;
    this.iouThreshold = 0.2; // Added NMS threshold

    // Models input sizes. Based on NDL standard shapes:
    this.deimSize = 800; // should in theory be 1024 but onnx complains if its not 800

    // Typical parseq model shapes (Width x Height) for ndl-lite weights
    this.parseq30Size = { w: 256, h: 16 };
    this.parseq50Size = { w: 384, h: 16 };
    this.parseq100Size = { w: 768, h: 16 };
  }

  /**
   * Initializes the ONNX models and loads the character list.
   */
  async init(config) {
    const options = config.ortOptions || { executionProviders: ["wasm"] };

    console.log("[INFO] Initializing Models...");

    [
      this.deimSession,
      this.parseq30Session,
      this.parseq50Session,
      this.parseq100Session,
    ] = await Promise.all([
      ort.InferenceSession.create(config.deimPath, options),
      ort.InferenceSession.create(config.parseq30Path, options),
      ort.InferenceSession.create(config.parseq50Path, options),
      ort.InferenceSession.create(config.parseq100Path, options),
    ]);

    this.charlist = config.charlist;
    console.log("[INFO] Initialization Complete");
  }

  /**
   * Main inference process equivalent to process() in ocr.py
   */
  async process(imageBlob) {
    const start = performance.now();

    // 1. Load image onto a canvas
    const imgBitmap = await createImageBitmap(imageBlob);
    const origW = imgBitmap.width;
    const origH = imgBitmap.height;
    const imgName = imageBlob.name || "image.png";

    const canvasOrig = document.createElement("canvas");
    canvasOrig.width = origW;
    canvasOrig.height = origH;
    const ctxOrig = canvasOrig.getContext("2d");
    ctxOrig.drawImage(imgBitmap, 0, 0);

    // 2. Preprocess and Run DEIM (Detector)
    console.log("[INFO] Running Detector...");
    const deimInputs = this._preprocessDEIM(canvasOrig, origW, origH);
    const deimOutputs = await this.deimSession.run(deimInputs);
    // 3. Extract and Apply Non-Maximum Suppression (NMS)
    const detections = this._postprocessDEIM(
      deimOutputs,
      Math.max(origW, origH),
    );
    console.log({ deimInputs, deimOutputs, detections });

    // 4. Calculate Reading Order (XY-Cut algorithm)
    console.log("[INFO] Calculating Reading Order (XY-Cut)...");
    if (detections.length > 0) {
      const bboxesCoords = detections.map((det) => det.box);
      const ranks = solveXYCut(bboxesCoords);

      // Attach the resulting rank to each detection and sort
      detections.forEach((det, i) => {
        det.rank = ranks[i];
      });
      detections.sort((a, b) => a.rank - b.rank);
    }

    // 5. Crop Line Regions and Process Cascade (Recognizer)
    console.log("[INFO] Running Recognizer Cascade...");
    const results = await this._processCascade(detections, canvasOrig);
    // 6. Construct JSON Output
    const resjsonarray = [];
    results.forEach((r, idx) => {
      // Skip empty results caused by impossibly small boxes
      if (!r.text) return;

      const [xmin, ymin, xmax, ymax] = r.box;
      const line_w = xmax - xmin;
      const line_h = ymax - ymin;

      resjsonarray.push({
        boundingBox: [
          [xmin, ymin],
          [xmin, ymin + line_h],
          [xmin + line_w, ymin],
          [xmin + line_w, ymin + line_h],
        ],
        id: idx,
        isVertical: r.isVertical ? "true" : "false",
        text: r.text,
        isTextline: "true",
        confidence: r.confidence,
        class: r.class,
        class_index: r.class_index,
      });
    });

    const calcTime = (performance.now() - start) / 1000;
    console.log(`[INFO] Total calculation time: ${calcTime.toFixed(3)} s`);

    return {
      contents: resjsonarray,
      imginfo: {
        img_width: origW,
        img_height: origH,
        img_path: imgName,
        img_name: imgName,
      },
    };
  }

  _preprocessDEIM(canvasOrig, origW, origH) {
    const maxWH = Math.max(origW, origH);
    const paddedCanvas = document.createElement("canvas");
    paddedCanvas.width = maxWH;
    paddedCanvas.height = maxWH;
    const pCtx = paddedCanvas.getContext("2d");
    pCtx.drawImage(canvasOrig, 0, 0);

    const resizeCanvas = document.createElement("canvas");
    resizeCanvas.width = this.deimSize;
    resizeCanvas.height = this.deimSize;
    const rCtx = resizeCanvas.getContext("2d");
    rCtx.drawImage(paddedCanvas, 0, 0, this.deimSize, this.deimSize);

    const imgData = rCtx.getImageData(0, 0, this.deimSize, this.deimSize).data;
    const floatData = new Float32Array(3 * this.deimSize * this.deimSize);

    const mean = [0.485, 0.456, 0.406];
    const std = [0.229, 0.224, 0.225];

    for (let i = 0; i < this.deimSize * this.deimSize; i++) {
      floatData[i] = (imgData[i * 4] / 255.0 - mean[0]) / std[0]; // R
      floatData[this.deimSize * this.deimSize + i] =
        (imgData[i * 4 + 1] / 255.0 - mean[1]) / std[1]; // G
      floatData[2 * this.deimSize * this.deimSize + i] =
        (imgData[i * 4 + 2] / 255.0 - mean[2]) / std[2]; // B
    }

    const tensorImage = new ort.Tensor("float32", floatData, [
      1,
      3,
      this.deimSize,
      this.deimSize,
    ]);
    const tensorSizes = new ort.Tensor(
      "int64",
      new BigInt64Array([BigInt(this.deimSize), BigInt(this.deimSize)]),
      [1, 2],
    );

    const inputs = {};
    inputs[this.deimSession.inputNames[0]] = tensorImage;
    inputs[this.deimSession.inputNames[1]] = tensorSizes;
    return inputs;
  }

  _postprocessDEIM(outputs, maxWH) {
    const outKeys = this.deimSession.outputNames;

    // DEIM usually outputs: class_ids, bboxes, scores, char_counts
    const classIdsT = outputs[outKeys[0]].data;
    const bboxesT = outputs[outKeys[1]].data;
    const scoresT = outputs[outKeys[2]].data;
    const charCountsT = outKeys.length >= 4 ? outputs[outKeys[3]].data : null;

    const detections = [];
    const scale = maxWH / this.deimSize;

    for (let i = 0; i < scoresT.length; i++) {
      if (scoresT[i] > this.detConfThreshold) {
        // bboxesT format is [x1, y1, x2, y2]
        let x1 = Math.round(bboxesT[i * 4] * scale);
        let y1 = Math.round(bboxesT[i * 4 + 1] * scale);
        let x2 = Math.round(bboxesT[i * 4 + 2] * scale);
        let y2 = Math.round(bboxesT[i * 4 + 3] * scale);
        let classIdx = Number(classIdsT[i]) - 1;
        detections.push({
          box: [x1, y1, x2, y2],
          confidence: scoresT[i],
          class_index: classIdx,
          class: classes[classIdx],
          pred_char_count: charCountsT ? Number(charCountsT[i]) : 100.0,
        });
      }
    }
    return detections;
  }
  async _processCascade(detections, canvasOrig) {
    const results = [];
    for (let i = 0; i < detections.length; i++) {
      const det = detections[i];
      const [x1, y1, x2, y2] = det.box;

      const cx1 = Math.max(0, Math.floor(x1));
      const cy1 = Math.max(0, Math.floor(y1));
      const cx2 = Math.min(canvasOrig.width, Math.ceil(x2));
      const cy2 = Math.min(canvasOrig.height, Math.ceil(y2));

      const cropW = cx2 - cx1;
      const cropH = cy2 - cy1;

      // Filter out invalid or impossibly small noise boxes
      if (cropW <= 5 || cropH <= 5) {
        results.push({ ...det, text: "", isVertical: false });
        continue;
      }

      const isVertical = cropH > cropW;
      const cropCanvas = document.createElement("canvas");

      if (isVertical) {
        cropCanvas.width = cropH;
        cropCanvas.height = cropW;
      } else {
        cropCanvas.width = cropW;
        cropCanvas.height = cropH;
      }

      const ctxCrop = cropCanvas.getContext("2d");
      if (isVertical) {
        ctxCrop.translate(0, cropW);
        ctxCrop.rotate(-Math.PI / 2);
      }
      ctxCrop.drawImage(canvasOrig, cx1, cy1, cropW, cropH, 0, 0, cropW, cropH);

      let text = "";
      let charCnt = Math.round(det.pred_char_count);

      if (charCnt === 3) {
        text = await this._runParseq(
          this.parseq30Session,
          cropCanvas,
          this.parseq30Size,
        );
        if (text.length >= 25) {
          text = await this._runParseq(
            this.parseq50Session,
            cropCanvas,
            this.parseq50Size,
          );
          if (text.length >= 45) {
            text = await this._runParseq(
              this.parseq100Session,
              cropCanvas,
              this.parseq100Size,
            );
          }
        }
      } else if (charCnt === 2) {
        text = await this._runParseq(
          this.parseq50Session,
          cropCanvas,
          this.parseq50Size,
        );
        if (text.length >= 45) {
          text = await this._runParseq(
            this.parseq100Session,
            cropCanvas,
            this.parseq100Size,
          );
        }
      } else {
        text = await this._runParseq(
          this.parseq100Session,
          cropCanvas,
          this.parseq100Size,
        );
      }

      results.push({ text, isVertical, ...det });
    }
    return results;
  }

  async _runParseq(session, cropCanvas, targetSize) {
    const resizeCanvas = document.createElement("canvas");
    resizeCanvas.width = targetSize.w;
    resizeCanvas.height = targetSize.h;
    const ctx = resizeCanvas.getContext("2d");
    ctx.drawImage(cropCanvas, 0, 0, targetSize.w, targetSize.h);

    const imgData = ctx.getImageData(0, 0, targetSize.w, targetSize.h).data;
    const floatData = new Float32Array(3 * targetSize.w * targetSize.h);

    for (let i = 0; i < targetSize.w * targetSize.h; i++) {
      floatData[i] = imgData[i * 4] / 127.5 - 1.0; // R
      floatData[targetSize.w * targetSize.h + i] =
        imgData[i * 4 + 1] / 127.5 - 1.0; // G
      floatData[2 * targetSize.w * targetSize.h + i] =
        imgData[i * 4 + 2] / 127.5 - 1.0; // B
    }

    const tensor = new ort.Tensor("float32", floatData, [
      1,
      3,
      targetSize.h,
      targetSize.w,
    ]);
    const inputs = {};
    inputs[session.inputNames[0]] = tensor;

    const outputMap = await session.run(inputs);
    const outTensor = outputMap[session.outputNames[0]];

    const dims = outTensor.dims;
    const data = outTensor.data;

    const seqLen = dims.length === 3 ? dims[1] : dims[0];
    const numClasses = dims.length === 3 ? dims[2] : dims[1];

    let resultStr = "";
    for (let i = 0; i < seqLen; i++) {
      let maxVal = -Infinity;
      let maxIdx = -1;

      for (let j = 0; j < numClasses; j++) {
        const val = data[i * numClasses + j];
        if (val > maxVal) {
          maxVal = val;
          maxIdx = j;
        }
      }

      if (maxIdx === 0) break; // Stop token
      if (maxIdx > 0 && maxIdx <= this.charlist.length) {
        resultStr += this.charlist[maxIdx - 1];
      }
    }
    return resultStr;
  }
}
