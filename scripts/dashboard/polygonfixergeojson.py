import geopandas as gpd

# Load
gdf = gpd.read_file("data/geo/uk_itl1_2025_bgc.geojson")

# Ensure WGS84
if gdf.crs is None or gdf.crs.to_epsg() != 4326:
    gdf = gdf.to_crs(epsg=4326)

# Validate geometries (fix invalid, but don't kill them)
gdf["geometry"] = gdf["geometry"].buffer(0)

# Explode multipolygons *and regroup by ITL1 code*
gdf = gdf.explode(ignore_index=True)
gdf = gdf.dissolve(by="ITL125CD", as_index=False)  # replace colname with your ITL1 ID col

# Simplify — use smaller tolerance for ITL1
gdf["geometry"] = gdf["geometry"].simplify(0.001, preserve_topology=True)

# Save
gdf.to_file("data/geo/ITL1_simplified_clean.geojson", driver="GeoJSON")
