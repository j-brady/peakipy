var _a;
import tz from "timezone";
import { AbstractSlider, AbstractRangeSliderView } from "./abstract_slider";
import { isString } from "../../core/util/types";
export class DatetimeRangeSliderView extends AbstractRangeSliderView {
}
DatetimeRangeSliderView.__name__ = "DatetimeRangeSliderView";
export class DatetimeRangeSlider extends AbstractSlider {
    constructor(attrs) {
        super(attrs);
        this.behaviour = "drag";
        this.connected = [false, true, false];
    }
    _formatter(value, format) {
        if (isString(format))
            return tz(value, format);
        else
            return format.compute(value);
    }
}
_a = DatetimeRangeSlider;
DatetimeRangeSlider.__name__ = "DatetimeRangeSlider";
(() => {
    _a.prototype.default_view = DatetimeRangeSliderView;
    _a.override({
        format: "%d %b %Y %H:%M:%S",
        step: 3600000, // 1 hour.
    });
})();
//# sourceMappingURL=datetime_range_slider.js.map