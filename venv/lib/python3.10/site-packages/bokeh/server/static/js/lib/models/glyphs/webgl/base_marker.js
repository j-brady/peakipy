import { BaseGLGlyph } from "./base";
import { Float32Buffer, NormalizedUint8Buffer, Uint8Buffer } from "./buffer";
import { marker_type_to_size_hint } from "./webgl_utils";
// Abstract base class for markers. All markers share the same GLSL, except for
// one function in the fragment shader that defines the marker geometry and is
// enabled through a #define.
export class BaseMarkerGL extends BaseGLGlyph {
    constructor(regl_wrapper, glyph) {
        super(regl_wrapper, glyph);
        this.glyph = glyph;
        this._antialias = 1.5;
        this._show_all = false;
    }
    _draw_one_marker_type(marker_type, transform, main_gl_glyph) {
        const props_no_hatch = {
            scissor: this.regl_wrapper.scissor,
            viewport: this.regl_wrapper.viewport,
            canvas_size: [transform.width, transform.height],
            pixel_ratio: transform.pixel_ratio,
            center: main_gl_glyph._centers,
            width: main_gl_glyph._widths,
            height: main_gl_glyph._heights,
            angle: main_gl_glyph._angles,
            size_hint: marker_type_to_size_hint(marker_type),
            nmarkers: main_gl_glyph.nvertices,
            antialias: this._antialias,
            linewidth: this._linewidths,
            line_color: this._line_rgba,
            fill_color: this._fill_rgba,
            line_cap: this._line_caps,
            line_join: this._line_joins,
            show: this._show,
        };
        if (this._have_hatch) {
            const props_hatch = {
                ...props_no_hatch,
                hatch_pattern: this._hatch_patterns,
                hatch_scale: this._hatch_scales,
                hatch_weight: this._hatch_weights,
                hatch_color: this._hatch_rgba,
            };
            this.regl_wrapper.marker_hatch(marker_type)(props_hatch);
        }
        else {
            this.regl_wrapper.marker_no_hatch(marker_type)(props_no_hatch);
        }
    }
    _set_visuals() {
        const visuals = this._get_visuals();
        const fill = visuals.fill;
        const line = visuals.line;
        if (this._linewidths == null) {
            // Either all or none are set.
            this._linewidths = new Float32Buffer(this.regl_wrapper);
            this._line_caps = new Uint8Buffer(this.regl_wrapper);
            this._line_joins = new Uint8Buffer(this.regl_wrapper);
            this._line_rgba = new NormalizedUint8Buffer(this.regl_wrapper);
            this._fill_rgba = new NormalizedUint8Buffer(this.regl_wrapper);
        }
        this._linewidths.set_from_prop(line.line_width);
        this._line_caps.set_from_line_cap(line.line_cap);
        this._line_joins.set_from_line_join(line.line_join);
        this._line_rgba.set_from_color(line.line_color, line.line_alpha);
        this._fill_rgba.set_from_color(fill.fill_color, fill.fill_alpha);
        this._have_hatch = visuals.hatch.doit;
        if (this._have_hatch) {
            const hatch = visuals.hatch;
            if (this._hatch_patterns == null) {
                // Either all or none are set.
                this._hatch_patterns = new Uint8Buffer(this.regl_wrapper);
                this._hatch_scales = new Float32Buffer(this.regl_wrapper);
                this._hatch_weights = new Float32Buffer(this.regl_wrapper);
                this._hatch_rgba = new NormalizedUint8Buffer(this.regl_wrapper);
            }
            this._hatch_patterns.set_from_hatch_pattern(hatch.hatch_pattern);
            this._hatch_scales.set_from_prop(hatch.hatch_scale);
            this._hatch_weights.set_from_prop(hatch.hatch_weight);
            this._hatch_rgba.set_from_color(hatch.hatch_color, hatch.hatch_alpha);
        }
    }
}
BaseMarkerGL.__name__ = "BaseMarkerGL";
// Avoiding use of nan or inf to represent missing data in webgl as shaders may
// have reduced floating point precision.  So here using a large-ish negative
// value instead.
BaseMarkerGL.missing_point = -10000;
//# sourceMappingURL=base_marker.js.map