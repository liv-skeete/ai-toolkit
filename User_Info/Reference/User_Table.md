# User Table

| Column Name        | Data Type    | Constraints         | Description              |
|--------------------|--------------|---------------------|--------------------------|
| id                 | String       | PRIMARY KEY         | Unique identifier        |
| username           | String(50)   | nullable            | User's unique username   |
| name               | String       | -                   | User's name              |
| email              | String       | -                   | User's email             |
| role               | String       | -                   | User's role              |
| profile_image_url  | Text         | -                   | Profile image path       |
| bio                | Text         | nullable            | User's biography         |
| gender             | Text         | nullable            | User's gender            |
| date_of_birth      | Date         | nullable            | User's date of birth     |
| last_active_at     | BigInteger   | -                   | Last activity timestamp  |
| updated_at         | BigInteger   | -                   | Last update timestamp    |
| created_at         | BigInteger   | -                   | Creation timestamp       |
| api_key            | String       | UNIQUE, nullable    | API authentication key   |
| settings           | JSON         | nullable            | User preferences         |
| info               | JSON         | nullable            | Additional user info     |
| oauth_sub          | Text         | UNIQUE              | OAuth subject identifier |