export interface Point { x: number; y: number; }
export interface UVPoint { u: number; v: number; }

/**
 * Generates a high-density (NxM) triangle mesh warp natively on the Canvas context.
 * It takes a base 2D bounding box and applies fluid, organic 3D-like curves to it
 * (like a waist pinch and a cylindrical chest bulge).
 */
export function drawDenseWarp(
  ctx: CanvasRenderingContext2D,
  img: CanvasImageSource, // e.g., HTMLImageElement
  imgW: number,
  imgH: number,
  startX: number,
  startY: number,
  width: number,
  height: number,
  pinchAmount: number,
  chestBulge: number
) {
  // A 12x12 grid creates 144 independent quads (288 triangles) for ultra-smooth curves
  const cols = 12;
  const rows = 12;

  const getPt = (c: number, r: number) => {
    const u = c / cols;
    const v = r / rows;

    // Base rectangular mapping
    let bx = startX + u * width;
    const by = startY + v * height;

    // Organic Deformation 1: Waist Pinch
    // It creates an hourglass-like curve peaking around the mid-to-lower abdomen
    let pinchV = Math.sin((v - 0.1) * Math.PI);
    if (pinchV < 0) pinchV = 0;
    // (1 - 2*u) means it shifts RIGHT on the left edge, and LEFT on the right edge.
    const inwardX = pinchV * pinchAmount * (1 - 2 * u);

    // Organic Deformation 2: Chest Bulge (Cylindrical 3D wrap effect)
    // Curves the horizontal spans downwards slightly in the center to wrap around a body
    const curveY = Math.sin(u * Math.PI) * Math.sin(v * Math.PI) * chestBulge;

    return {
      x: bx + inwardX,
      y: by + curveY,
      u: u * imgW,
      v: v * imgH
    };
  };

  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const p0 = getPt(c, r);
      const p1 = getPt(c + 1, r);
      const p2 = getPt(c, r + 1);
      const p3 = getPt(c + 1, r + 1);

      // Split quad into two triangles
      drawExpandedTriangle(ctx, img, p0, p1, p2);
      drawExpandedTriangle(ctx, img, p1, p3, p2);
    }
  }
}

/**
 * Renders an affine-transformed image within a triangle clip path.
 * Centroid-based sub-pixel expansion eliminates rendering "seams" between triangles.
 */
function drawExpandedTriangle(ctx: CanvasRenderingContext2D, img: CanvasImageSource, p0: any, p1: any, p2: any) {
  const cx = (p0.x + p1.x + p2.x) / 3;
  const cy = (p0.y + p1.y + p2.y) / 3;

  // Sub-pixel expansion to perfectly overwrite adjacent transparent seams from HTML5 clipping antialiasing
  const exp = 0.6;
  const len0 = Math.hypot(p0.x - cx, p0.y - cy) || 1;
  const len1 = Math.hypot(p1.x - cx, p1.y - cy) || 1;
  const len2 = Math.hypot(p2.x - cx, p2.y - cy) || 1;

  const e0 = { x: p0.x + ((p0.x - cx) / len0) * exp, y: p0.y + ((p0.y - cy) / len0) * exp };
  const e1 = { x: p1.x + ((p1.x - cx) / len1) * exp, y: p1.y + ((p1.y - cy) / len1) * exp };
  const e2 = { x: p2.x + ((p2.x - cx) / len2) * exp, y: p2.y + ((p2.y - cy) / len2) * exp };

  ctx.save();
  ctx.beginPath();
  ctx.moveTo(e0.x, e0.y);
  ctx.lineTo(e1.x, e1.y);
  ctx.lineTo(e2.x, e2.y);
  ctx.closePath();
  ctx.clip(); // Restrict drawing strictly to this precise triangle

  const u0 = p0.u, v0 = p0.v;
  const u1 = p1.u, v1 = p1.v;
  const u2 = p2.u, v2 = p2.v;

  const det = (u0 - u2) * (v1 - v2) - (u1 - u2) * (v0 - v2);
  if (det === 0) {
    ctx.restore();
    return;
  }

  // Calculate Affine Transformation Matrix
  const A = ((e0.x - e2.x) * (v1 - v2) - (e1.x - e2.x) * (v0 - v2)) / det;
  const B = ((u0 - u2) * (e1.x - e2.x) - (u1 - u2) * (e0.x - e2.x)) / det;
  const C = ((e0.y - e2.y) * (v1 - v2) - (e1.y - e2.y) * (v0 - v2)) / det;
  const D = ((u0 - u2) * (e1.y - e2.y) - (u1 - u2) * (e0.y - e2.y)) / det;
  const E = e2.x - A * u2 - B * v2;
  const F = e2.y - C * u2 - D * v2;

  ctx.transform(A, C, B, D, E, F);
  ctx.drawImage(img, 0, 0);
  ctx.restore();
}
