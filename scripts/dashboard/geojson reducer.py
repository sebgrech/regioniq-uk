import geopandas as gpd
gdf = gpd.read_file("data/geo/uk_itl1_2025_bgc.geojson")
gdf["geometry"] = gdf["geometry"].simplify(tolerance=500, preserve_topology=True)
gdf.to_file("data/geo/ITL1_simplified.geojson", driver="GeoJSON")