-- =============================================================================
-- 06_state_centroids.sql — Average coordinates per state (for the dashboard map)
-- The geolocation table has some points outside Brazil (recording errors),
-- so we keep only points inside Brazil's bounding box before averaging.
-- =============================================================================

SELECT
    geolocation_state               AS state,
    AVG(geolocation_lat)            AS lat,
    AVG(geolocation_lng)            AS lng,
    COUNT(*)                        AS points
FROM geolocation
WHERE geolocation_lat BETWEEN -34.0 AND 5.3
  AND geolocation_lng BETWEEN -74.0 AND -34.7
GROUP BY geolocation_state
ORDER BY geolocation_state;
