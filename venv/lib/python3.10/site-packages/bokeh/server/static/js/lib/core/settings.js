export class Settings {
    constructor() {
        this._dev = false;
        this._wireframe = false;
        this._force_webgl = false;
    }
    set dev(dev) {
        this._dev = dev;
    }
    get dev() {
        return this._dev;
    }
    set wireframe(wireframe) {
        this._wireframe = wireframe;
    }
    get wireframe() {
        return this._wireframe;
    }
    set force_webgl(force_webgl) {
        this._force_webgl = force_webgl;
    }
    get force_webgl() {
        return this._force_webgl;
    }
}
Settings.__name__ = "Settings";
export const settings = new Settings();
//# sourceMappingURL=settings.js.map