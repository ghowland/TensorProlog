// ============================================================
// src/builtins/register_mappings.zig
// ============================================================

//NOTE(geoff): This file was not in the original generation, started in turn 21/40

const dispatch_reg = @import("dispatch.zig");
const BuiltinTable = dispatch_reg.BuiltinTable;
const mappings_mod = @import("mappings.zig");
const conversion_mod = @import("conversion.zig");

pub fn registerMappingBuiltins(table: *BuiltinTable) void {
    table.register(200, "map_get", mappings_mod.builtinMapGet, true, 1, .value);
    table.register(201, "map_set", mappings_mod.builtinMapSet, true, 2, .value);
    table.register(202, "map_delete", mappings_mod.builtinMapDelete, true, 1, .boolean);
    table.register(203, "map_contains_key", mappings_mod.builtinMapContainsKey, true, 1, .boolean);
    table.register(204, "map_keys", mappings_mod.builtinMapKeys, true, 0, .value);
    table.register(205, "map_values", mappings_mod.builtinMapValues, true, 0, .value);
    table.register(206, "map_size", mappings_mod.builtinMapSize, true, 0, .value);
    table.register(207, "map_merge", mappings_mod.builtinMapMerge, true, 0, .value);
    table.register(208, "map_filter_keys", mappings_mod.builtinMapFilterKeys, true, 0, .value);
    table.register(209, "map_filter_values", mappings_mod.builtinMapFilterValues, true, 0, .value);
    table.register(210, "map_map_values", mappings_mod.builtinMapMapValues, true, 0, .value);
    table.register(211, "map_invert", mappings_mod.builtinMapInvert, true, 0, .value);
    table.register(212, "map_clear", mappings_mod.builtinMapClear, true, 0, .empty);
    table.register(213, "map_equal", mappings_mod.builtinMapEqual, true, 0, .boolean);
    table.register(214, "map_from_arrays", mappings_mod.builtinMapFromArrays, true, 0, .value);
}

pub fn registerConversionBuiltins(table: *BuiltinTable) void {
    table.register(300, "parse_json", conversion_mod.builtinParseJson, false, 1, .empty);
    table.register(301, "parse_csv", conversion_mod.builtinParseCsv, false, 1, .empty);
    table.register(302, "parse_xml", conversion_mod.builtinParseXml, false, 1, .empty);
    table.register(303, "parse_yaml", conversion_mod.builtinParseYaml, false, 1, .empty);
    table.register(304, "to_json", conversion_mod.builtinToJson, true, 0, .text);
    table.register(305, "to_csv", conversion_mod.builtinToCsv, true, 0, .text);
    table.register(306, "to_fraction", conversion_mod.builtinToFraction, true, 0, .value);
    table.register(307, "from_fraction", conversion_mod.builtinFromFraction, true, 1, .value);
    table.register(308, "vdr_to_decimal", conversion_mod.builtinVdrToDecimalString, true, 1, .text);
    table.register(309, "decimal_to_vdr", conversion_mod.builtinDecimalStringToVdr, true, 1, .value);
    table.register(310, "base_convert", conversion_mod.builtinBaseConvert, true, 0, .text);
    table.register(311, "timestamp_fields", conversion_mod.builtinTimestampToFields, true, 0, .value);
}
