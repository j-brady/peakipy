import { BaseMarkerGL } from "./base_marker";
import { Uint8Buffer } from "./buffer";
export class SingleMarkerGL extends BaseMarkerGL {
    constructor(regl_wrapper, glyph) {
        super(regl_wrapper, glyph);
        this.glyph = glyph;
    }
    _draw_impl(indices, transform, main_gl_glyph, marker_type) {
        if (main_gl_glyph.data_changed) {
            main_gl_glyph._set_data();
            main_gl_glyph.data_changed = false;
        }
        if (this.visuals_changed) {
            this._set_visuals();
            this.visuals_changed = false;
        }
        const nmarkers = main_gl_glyph.nvertices;
        if (this._show == null)
            this._show = new Uint8Buffer(this.regl_wrapper);
        const prev_nmarkers = this._show.length;
        const show_array = this._show.get_sized_array(nmarkers);
        if (indices.length < nmarkers) {
            this._show_all = false;
            // Reset all show values to zero.
            for (let i = 0; i < nmarkers; i++)
                show_array[i] = 0;
            // Set show values of markers to render to 255.
            for (let j = 0; j < indices.length; j++) {
                show_array[indices[j]] = 255;
            }
        }
        else if (!this._show_all || prev_nmarkers != nmarkers) {
            this._show_all = true;
            for (let i = 0; i < nmarkers; i++)
                show_array[i] = 255;
        }
        this._show.update();
        this._draw_one_marker_type(marker_type, transform, main_gl_glyph);
    }
}
SingleMarkerGL.__name__ = "SingleMarkerGL";
//# sourceMappingURL=single_marker.js.map