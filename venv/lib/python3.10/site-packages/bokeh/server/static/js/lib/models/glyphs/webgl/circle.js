import { Float32Buffer } from "./buffer";
import { SingleMarkerGL } from "./single_marker";
export class CircleGL extends SingleMarkerGL {
    constructor(regl_wrapper, glyph) {
        super(regl_wrapper, glyph);
        this.glyph = glyph;
    }
    draw(indices, main_glyph, transform) {
        this._draw_impl(indices, transform, main_glyph.glglyph, "circle");
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
            this._heights = this._widths;
            this._angles = new Float32Buffer(this.regl_wrapper);
        }
        const centers_array = this._centers.get_sized_array(nmarkers * 2);
        const widths_array = this._widths.get_sized_array(nmarkers);
        for (let i = 0; i < nmarkers; i++) {
            if (isFinite(this.glyph.sx[i]) && isFinite(this.glyph.sy[i])) {
                centers_array[2 * i] = this.glyph.sx[i];
                centers_array[2 * i + 1] = this.glyph.sy[i];
            }
            else {
                centers_array[2 * i] = SingleMarkerGL.missing_point;
                centers_array[2 * i + 1] = SingleMarkerGL.missing_point;
            }
            widths_array[i] = this.glyph.sradius[i] * 2;
        }
        this._centers.update();
        this._widths.update();
        this._angles.set_from_prop(this.glyph.angle);
    }
}
CircleGL.__name__ = "CircleGL";
//# sourceMappingURL=circle.js.map