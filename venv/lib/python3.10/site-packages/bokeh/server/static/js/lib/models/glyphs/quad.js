var _a;
import { Box, BoxView } from "./box";
import * as p from "../../core/properties";
export class QuadView extends BoxView {
    async lazy_initialize() {
        await super.lazy_initialize();
        const { webgl } = this.renderer.plot_view.canvas_view;
        if (webgl != null && webgl.regl_wrapper.has_webgl) {
            const { LRTBGL } = await import("./webgl/lrtb");
            this.glglyph = new LRTBGL(webgl.regl_wrapper, this);
        }
    }
    scenterxy(i) {
        const scx = this.sleft[i] / 2 + this.sright[i] / 2;
        const scy = this.stop[i] / 2 + this.sbottom[i] / 2;
        return [scx, scy];
    }
    _lrtb(i) {
        const l = this._left[i];
        const r = this._right[i];
        const t = this._top[i];
        const b = this._bottom[i];
        return [l, r, t, b];
    }
}
QuadView.__name__ = "QuadView";
export class Quad extends Box {
    constructor(attrs) {
        super(attrs);
    }
}
_a = Quad;
Quad.__name__ = "Quad";
(() => {
    _a.prototype.default_view = QuadView;
    _a.define(({}) => ({
        right: [p.XCoordinateSpec, { field: "right" }],
        bottom: [p.YCoordinateSpec, { field: "bottom" }],
        left: [p.XCoordinateSpec, { field: "left" }],
        top: [p.YCoordinateSpec, { field: "top" }],
    }));
})();
//# sourceMappingURL=quad.js.map