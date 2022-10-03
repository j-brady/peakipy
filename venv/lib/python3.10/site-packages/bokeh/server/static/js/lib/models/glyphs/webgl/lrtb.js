import { Float32Buffer } from "./buffer";
import { SingleMarkerGL } from "./single_marker";
export class LRTBGL extends SingleMarkerGL {
    constructor(regl_wrapper, glyph) {
        super(regl_wrapper, glyph);
        this.glyph = glyph;
    }
    draw(indices, main_glyph, transform) {
        this._draw_impl(indices, transform, main_glyph.glglyph, "square");
    }
    _get_visuals() {
        return this.glyph.visuals;
    }
    _set_data() {
        const nmarkers = this.nvertices;
        if (this._centers == null) {
            // Either all or none are set.
            this._centers = new Float32Buffer(this.regl_wrapper);
            this._widths = new Float32Buffer(this.regl_wrapper);
            this._heights = new Float32Buffer(this.regl_wrapper);
            this._angles = new Float32Buffer(this.regl_wrapper);
            this._angles.set_from_scalar(0);
        }
        const centers_array = this._centers.get_sized_array(nmarkers * 2);
        const heights_array = this._heights.get_sized_array(nmarkers);
        const widths_array = this._widths.get_sized_array(nmarkers);
        for (let i = 0; i < nmarkers; i++) {
            const l = this.glyph.sleft[i];
            const r = this.glyph.sright[i];
            const t = this.glyph.stop[i];
            const b = this.glyph.sbottom[i];
            if (isFinite(l) && isFinite(r) && isFinite(t) && isFinite(b)) {
                centers_array[2 * i] = (l + r) / 2;
                centers_array[2 * i + 1] = (t + b) / 2;
                heights_array[i] = Math.abs(t - b);
                widths_array[i] = Math.abs(r - l);
            }
            else {
                centers_array[2 * i] = SingleMarkerGL.missing_point;
                centers_array[2 * i + 1] = SingleMarkerGL.missing_point;
                heights_array[i] = SingleMarkerGL.missing_point;
                widths_array[i] = SingleMarkerGL.missing_point;
            }
        }
        this._centers.update();
        this._heights.update();
        this._widths.update();
    }
}
LRTBGL.__name__ = "LRTBGL";
//# sourceMappingURL=lrtb.js.map