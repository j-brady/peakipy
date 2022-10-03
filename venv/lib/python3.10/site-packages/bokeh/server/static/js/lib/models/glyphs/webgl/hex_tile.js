import { Float32Buffer } from "./buffer";
import { SingleMarkerGL } from "./single_marker";
export class HexTileGL extends SingleMarkerGL {
    constructor(regl_wrapper, glyph) {
        super(regl_wrapper, glyph);
        this.glyph = glyph;
    }
    draw(indices, main_glyph, transform) {
        this._draw_impl(indices, transform, main_glyph.glglyph, "hex");
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
        }
        const centers_array = this._centers.get_sized_array(nmarkers * 2);
        for (let i = 0; i < nmarkers; i++) {
            if (isFinite(this.glyph.sx[i]) && isFinite(this.glyph.sy[i])) {
                centers_array[2 * i] = this.glyph.sx[i];
                centers_array[2 * i + 1] = this.glyph.sy[i];
            }
            else {
                centers_array[2 * i] = SingleMarkerGL.missing_point;
                centers_array[2 * i + 1] = SingleMarkerGL.missing_point;
            }
        }
        this._centers.update();
        if (this.glyph.model.orientation == "pointytop") {
            this._angles.set_from_scalar(0.5 * Math.PI);
            this._widths.set_from_scalar(this.glyph.svy[0] * 2);
            this._heights.set_from_scalar(this.glyph.svx[4] * 4 / Math.sqrt(3));
        }
        else {
            this._angles.set_from_scalar(0);
            this._widths.set_from_scalar(this.glyph.svx[0] * 2);
            this._heights.set_from_scalar(this.glyph.svy[4] * 4 / Math.sqrt(3));
        }
    }
}
HexTileGL.__name__ = "HexTileGL";
//# sourceMappingURL=hex_tile.js.map