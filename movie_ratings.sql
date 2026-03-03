CREATE TABLE IF NOT EXISTS movie_ratings (
    userId INTEGER,
    movieId INTEGER,
    title VARCHAR(255),
    year INTEGER,
    genres VARCHAR(255),
    rating DECIMAL(2,1),
    timestamp BIGINT,
    director VARCHAR(255),
    popularity DECIMAL(10,2),
    vote_count INTEGER,
    overview TEXT
);

-- USER COHORT ANALYSIS 

SELECT 
    CASE 
        WHEN rating_count >= 15 THEN 'Power User'
        ELSE 'Casual User'
    END AS user_segment,
    COUNT(DISTINCT userId) as total_users,
    ROUND(AVG(avg_rating), 2) as group_avg_rating
FROM (
    SELECT 
        userId, 
        COUNT(movieId) as rating_count,
        AVG(rating) as avg_rating
    FROM movie_ratings
    GROUP BY userId
) as user_stats
GROUP BY 1;

-- CONTENT PERFORMANCE ANALYSIS

SELECT 
    title,
    director,
    rating,
    popularity,
    CASE
        WHEN rating > 4.0 AND popularity < 80 THEN 'Hidden Gem'
        WHEN rating < 3.0 AND popularity > 100 THEN 'Overhyped'
        ELSE 'Standard'
    END as movie_category
FROM movie_ratings
WHERE vote_count > 500
ORDER BY rating DESC
LIMIT 10;

-- GENRE ENGAGEMENT

SELECT 
    'Action' as genre, AVG(rating) as avg_score, COUNT(*) as count FROM movie_ratings WHERE genres LIKE '%Action%'
UNION ALL
SELECT 
    'Drama' as genre, AVG(rating) as avg_score, COUNT(*) as count FROM movie_ratings WHERE genres LIKE '%Drama%'
UNION ALL
SELECT 
    'Comedy' as genre, AVG(rating) as avg_score, COUNT(*) as count FROM movie_ratings WHERE genres LIKE '%Comedy%'
UNION ALL
SELECT 
    'Sci-Fi' as genre, AVG(rating) as avg_score, COUNT(*) as count FROM movie_ratings WHERE genres LIKE '%Sci-Fi%'
ORDER BY avg_score DESC;