CREATE TABLE vehicle_class (
    vehicle_id uuid DEFAULT gen_random_uuid(),
    vehicle_name VARCHAR(255) NOT NULL,
    PRIMARY KEY (vehicle_id)
);

CREATE TABLE vehicle_count (
  vehicle_count_id uuid DEFAULT gen_random_uuid(),
  class_id    uuid NOT NULL REFERENCES vehicle_class(vehicle_id) ON DELETE CASCADE,
  period_date DATE NOT NULL,
  count       INTEGER NOT NULL CHECK (count >= 0),
  PRIMARY KEY (vehicle_count_id)
);

