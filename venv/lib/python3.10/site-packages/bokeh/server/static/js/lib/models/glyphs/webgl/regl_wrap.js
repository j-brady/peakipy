import createRegl from "regl";
import { DashCache } from "./dash_cache";
import line_vertex_shader from "./regl_line.vert";
import line_fragment_shader from "./regl_line.frag";
import marker_vertex_shader from "./marker.vert";
import marker_fragment_shader from "./marker.frag";
// All access to regl is performed via the get_regl() function that returns a
// ReglWrapper object.  This ensures that regl is correctly initialised before
// it is used, and is only initialised once.
let regl_wrapper;
export function get_regl(gl) {
    if (regl_wrapper == null)
        regl_wrapper = new ReglWrapper(gl);
    return regl_wrapper;
}
export class ReglWrapper {
    constructor(gl) {
        try {
            this._regl = createRegl({
                gl,
                extensions: [
                    "ANGLE_instanced_arrays",
                    "EXT_blend_minmax",
                ],
            });
            this._regl_available = true;
            // Initialise static Buffers/Elements.
            this._line_geometry = this._regl.buffer({
                usage: "static",
                type: "float",
                data: [[-2, 0], [-1, -1], [1, -1], [2, 0], [1, 1], [-1, 1]],
            });
            this._line_triangles = this._regl.elements({
                usage: "static",
                primitive: "triangles",
                data: [[0, 1, 5], [1, 2, 5], [5, 2, 4], [2, 3, 4]],
            });
        }
        catch (err) {
            this._regl_available = false;
        }
    }
    // Create and return ReGL Buffer.
    buffer(options) {
        return this._regl.buffer(options);
    }
    clear(width, height) {
        this._viewport = { x: 0, y: 0, width, height };
        this._regl.clear({ color: [0, 0, 0, 0] });
    }
    get has_webgl() {
        return this._regl_available;
    }
    get scissor() {
        return this._scissor;
    }
    set_scissor(x, y, width, height) {
        this._scissor = { x, y, width, height };
    }
    get viewport() {
        return this._viewport;
    }
    dashed_line() {
        if (this._dashed_line == null)
            this._dashed_line = regl_dashed_line(this._regl, this._line_geometry, this._line_triangles);
        return this._dashed_line;
    }
    get_dash(line_dash) {
        if (this._dash_cache == null)
            this._dash_cache = new DashCache(this._regl);
        return this._dash_cache.get(line_dash);
    }
    marker_no_hatch(marker_type) {
        if (this._marker_no_hatch_map == null)
            this._marker_no_hatch_map = new Map();
        let func = this._marker_no_hatch_map.get(marker_type);
        if (func == null) {
            func = regl_marker_no_hatch(this._regl, marker_type);
            this._marker_no_hatch_map.set(marker_type, func);
        }
        return func;
    }
    marker_hatch(marker_type) {
        if (this._marker_hatch_map == null)
            this._marker_hatch_map = new Map();
        let func = this._marker_hatch_map.get(marker_type);
        if (func == null) {
            func = regl_marker_hatch(this._regl, marker_type);
            this._marker_hatch_map.set(marker_type, func);
        }
        return func;
    }
    solid_line() {
        if (this._solid_line == null)
            this._solid_line = regl_solid_line(this._regl, this._line_geometry, this._line_triangles);
        return this._solid_line;
    }
}
ReglWrapper.__name__ = "ReglWrapper";
// Regl rendering functions are here as some will be reused, e.g. lines may also
// be used around polygons or for bezier curves.
// Mesh for line rendering (solid and dashed).
//
//   1       5-----4
//          /|\    |\
//         / | \   | \
// y 0    0  |  \  |  3
//         \ |   \ | /
//          \|    \|/
//  -1       1-----2
//
//       -2  -1    1  2
//              x
function regl_solid_line(regl, line_geometry, line_triangles) {
    const config = {
        vert: line_vertex_shader,
        frag: line_fragment_shader,
        attributes: {
            a_position: {
                buffer: line_geometry,
                divisor: 0,
            },
            a_point_prev(_, props) {
                return props.points.to_attribute_config();
            },
            a_point_start(_, props) {
                return props.points.to_attribute_config(Float32Array.BYTES_PER_ELEMENT * 2);
            },
            a_point_end(_, props) {
                return props.points.to_attribute_config(Float32Array.BYTES_PER_ELEMENT * 4);
            },
            a_point_next(_, props) {
                return props.points.to_attribute_config(Float32Array.BYTES_PER_ELEMENT * 6);
            },
            a_show_prev(_, props) {
                return props.show.to_attribute_config();
            },
            a_show_curr(_, props) {
                return props.show.to_attribute_config(Uint8Array.BYTES_PER_ELEMENT);
            },
            a_show_next(_, props) {
                return props.show.to_attribute_config(Uint8Array.BYTES_PER_ELEMENT * 2);
            },
        },
        uniforms: {
            u_canvas_size: regl.prop("canvas_size"),
            u_pixel_ratio: regl.prop("pixel_ratio"),
            u_antialias: regl.prop("antialias"),
            u_line_color: regl.prop("line_color"),
            u_linewidth: regl.prop("linewidth"),
            u_miter_limit: regl.prop("miter_limit"),
            u_line_join: regl.prop("line_join"),
            u_line_cap: regl.prop("line_cap"),
        },
        elements: line_triangles,
        instances: regl.prop("nsegments"),
        blend: {
            enable: true,
            equation: "max",
            func: {
                srcRGB: 1,
                srcAlpha: 1,
                dstRGB: 1,
                dstAlpha: 1,
            },
        },
        depth: { enable: false },
        scissor: {
            enable: true,
            box: regl.prop("scissor"),
        },
        viewport: regl.prop("viewport"),
    };
    return regl(config);
}
function regl_dashed_line(regl, line_geometry, line_triangles) {
    const config = {
        vert: `#define DASHED\n\n${line_vertex_shader}`,
        frag: `#define DASHED\n\n${line_fragment_shader}`,
        attributes: {
            a_position: {
                buffer: line_geometry,
                divisor: 0,
            },
            a_point_prev(_, props) {
                return props.points.to_attribute_config();
            },
            a_point_start(_, props) {
                return props.points.to_attribute_config(Float32Array.BYTES_PER_ELEMENT * 2);
            },
            a_point_end(_, props) {
                return props.points.to_attribute_config(Float32Array.BYTES_PER_ELEMENT * 4);
            },
            a_point_next(_, props) {
                return props.points.to_attribute_config(Float32Array.BYTES_PER_ELEMENT * 6);
            },
            a_show_prev(_, props) {
                return props.show.to_attribute_config();
            },
            a_show_curr(_, props) {
                return props.show.to_attribute_config(Uint8Array.BYTES_PER_ELEMENT);
            },
            a_show_next(_, props) {
                return props.show.to_attribute_config(Uint8Array.BYTES_PER_ELEMENT * 2);
            },
            a_length_so_far(_, props) {
                return props.length_so_far.to_attribute_config();
            },
        },
        uniforms: {
            u_canvas_size: regl.prop("canvas_size"),
            u_pixel_ratio: regl.prop("pixel_ratio"),
            u_antialias: regl.prop("antialias"),
            u_line_color: regl.prop("line_color"),
            u_linewidth: regl.prop("linewidth"),
            u_miter_limit: regl.prop("miter_limit"),
            u_line_join: regl.prop("line_join"),
            u_line_cap: regl.prop("line_cap"),
            u_dash_tex: regl.prop("dash_tex"),
            u_dash_tex_info: regl.prop("dash_tex_info"),
            u_dash_scale: regl.prop("dash_scale"),
            u_dash_offset: regl.prop("dash_offset"),
        },
        elements: line_triangles,
        instances: regl.prop("nsegments"),
        blend: {
            enable: true,
            equation: "max",
            func: {
                srcRGB: 1,
                srcAlpha: 1,
                dstRGB: 1,
                dstAlpha: 1,
            },
        },
        depth: { enable: false },
        scissor: {
            enable: true,
            box: regl.prop("scissor"),
        },
        viewport: regl.prop("viewport"),
    };
    return regl(config);
}
function regl_marker_no_hatch(regl, marker_type) {
    const config = {
        vert: marker_vertex_shader,
        frag: `#define USE_${marker_type.toUpperCase()}\n${marker_fragment_shader}`,
        attributes: {
            a_position: {
                buffer: regl.buffer([[-0.5, -0.5], [-0.5, 0.5], [0.5, 0.5], [0.5, -0.5]]),
                divisor: 0,
            },
            a_center(_, props) {
                return props.center.to_attribute_config();
            },
            a_width(_, props) {
                return props.width.to_attribute_config();
            },
            a_height(_, props) {
                return props.height.to_attribute_config();
            },
            a_angle(_, props) {
                return props.angle.to_attribute_config();
            },
            a_linewidth(_, props) {
                return props.linewidth.to_attribute_config();
            },
            a_line_color(_, props) {
                return props.line_color.to_attribute_config();
            },
            a_fill_color(_, props) {
                return props.fill_color.to_attribute_config();
            },
            a_line_cap(_, props) {
                return props.line_cap.to_attribute_config();
            },
            a_line_join(_, props) {
                return props.line_join.to_attribute_config();
            },
            a_show(_, props) {
                return props.show.to_attribute_config();
            },
        },
        uniforms: {
            u_canvas_size: regl.prop("canvas_size"),
            u_pixel_ratio: regl.prop("pixel_ratio"),
            u_antialias: regl.prop("antialias"),
            u_size_hint: regl.prop("size_hint"),
        },
        count: 4,
        primitive: "triangle fan",
        instances: regl.prop("nmarkers"),
        blend: {
            enable: true,
            func: {
                srcRGB: "one",
                srcAlpha: "one",
                dstRGB: "one minus src alpha",
                dstAlpha: "one minus src alpha",
            },
        },
        depth: { enable: false },
        scissor: {
            enable: true,
            box: regl.prop("scissor"),
        },
        viewport: regl.prop("viewport"),
    };
    return regl(config);
}
function regl_marker_hatch(regl, marker_type) {
    const config = {
        vert: `#define HATCH\n${marker_vertex_shader}`,
        frag: `#define USE_${marker_type.toUpperCase()}\n#define HATCH\n${marker_fragment_shader}`,
        attributes: {
            a_position: {
                buffer: regl.buffer([[-0.5, -0.5], [-0.5, 0.5], [0.5, 0.5], [0.5, -0.5]]),
                divisor: 0,
            },
            a_center(_, props) {
                return props.center.to_attribute_config();
            },
            a_width(_, props) {
                return props.width.to_attribute_config();
            },
            a_height(_, props) {
                return props.height.to_attribute_config();
            },
            a_angle(_, props) {
                return props.angle.to_attribute_config();
            },
            a_linewidth(_, props) {
                return props.linewidth.to_attribute_config();
            },
            a_line_color(_, props) {
                return props.line_color.to_attribute_config();
            },
            a_fill_color(_, props) {
                return props.fill_color.to_attribute_config();
            },
            a_line_cap(_, props) {
                return props.line_cap.to_attribute_config();
            },
            a_line_join(_, props) {
                return props.line_join.to_attribute_config();
            },
            a_show(_, props) {
                return props.show.to_attribute_config();
            },
            a_hatch_pattern(_, props) {
                return props.hatch_pattern.to_attribute_config();
            },
            a_hatch_scale(_, props) {
                return props.hatch_scale.to_attribute_config();
            },
            a_hatch_weight(_, props) {
                return props.hatch_weight.to_attribute_config();
            },
            a_hatch_color(_, props) {
                return props.hatch_color.to_attribute_config();
            },
        },
        uniforms: {
            u_canvas_size: regl.prop("canvas_size"),
            u_pixel_ratio: regl.prop("pixel_ratio"),
            u_antialias: regl.prop("antialias"),
            u_size_hint: regl.prop("size_hint"),
        },
        count: 4,
        primitive: "triangle fan",
        instances: regl.prop("nmarkers"),
        blend: {
            enable: true,
            func: {
                srcRGB: "one",
                srcAlpha: "one",
                dstRGB: "one minus src alpha",
                dstAlpha: "one minus src alpha",
            },
        },
        depth: { enable: false },
        scissor: {
            enable: true,
            box: regl.prop("scissor"),
        },
        viewport: regl.prop("viewport"),
    };
    return regl(config);
}
//# sourceMappingURL=regl_wrap.js.map