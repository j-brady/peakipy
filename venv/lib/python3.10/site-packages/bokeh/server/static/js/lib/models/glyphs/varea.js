var _a;
import { Area, AreaView } from "./area";
import * as hittest from "../../core/hittest";
import * as p from "../../core/properties";
import { Selection } from "../selections/selection";
export class VAreaView extends AreaView {
    _index_data(index) {
        const { min, max } = Math;
        const { data_size } = this;
        for (let i = 0; i < data_size; i++) {
            const x = this._x[i];
            const y1 = this._y1[i];
            const y2 = this._y2[i];
            index.add_rect(x, min(y1, y2), x, max(y1, y2));
        }
    }
    _render(ctx, _indices, data) {
        const { sx, sy1, sy2 } = data ?? this;
        ctx.beginPath();
        for (let i = 0, end = sy1.length; i < end; i++) {
            ctx.lineTo(sx[i], sy1[i]);
        }
        // iterate backwards so that the upper end is below the lower start
        for (let i = sy2.length - 1; i >= 0; i--) {
            ctx.lineTo(sx[i], sy2[i]);
        }
        ctx.closePath();
        this.visuals.fill.apply(ctx);
        this.visuals.hatch.apply(ctx);
    }
    scenterxy(i) {
        const scx = this.sx[i];
        const scy = (this.sy1[i] + this.sy2[i]) / 2;
        return [scx, scy];
    }
    _hit_point(geometry) {
        const L = this.sx.length;
        const result = new Selection();
        for (let i = 0, end = L - 1; i < end; i++) {
            const sx = [this.sx[i], this.sx[i + 1], this.sx[i + 1], this.sx[i]];
            const sy = [this.sy1[i], this.sy1[i + 1], this.sy2[i + 1], this.sy2[i]];
            if (hittest.point_in_poly(geometry.sx, geometry.sy, sx, sy)) {
                result.add_to_selected_glyphs(this.model);
                result.view = this;
                result.line_indices = [i];
                break;
            }
        }
        return result;
    }
    _map_data() {
        this.sx = this.renderer.xscale.v_compute(this._x);
        this.sy1 = this.renderer.yscale.v_compute(this._y1);
        this.sy2 = this.renderer.yscale.v_compute(this._y2);
    }
}
VAreaView.__name__ = "VAreaView";
export class VArea extends Area {
    constructor(attrs) {
        super(attrs);
    }
}
_a = VArea;
VArea.__name__ = "VArea";
(() => {
    _a.prototype.default_view = VAreaView;
    _a.define(({}) => ({
        x: [p.XCoordinateSpec, { field: "x" }],
        y1: [p.YCoordinateSpec, { field: "y1" }],
        y2: [p.YCoordinateSpec, { field: "y2" }],
    }));
})();
//# sourceMappingURL=varea.js.map