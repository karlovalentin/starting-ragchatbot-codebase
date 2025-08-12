"""Tests for VectorStore to diagnose vector database issues."""

import json
import tempfile
from unittest.mock import patch

from mock_data import SAMPLE_COURSES, create_mock_course
from models import CourseChunk
from test_utils import cleanup_test_files
from vector_store import VectorStore


class TestVectorStoreInitialization:
    """Test VectorStore initialization"""

    def test_vector_store_creation(self):
        """Test creating a VectorStore instance"""
        temp_dir = tempfile.mkdtemp()

        try:
            vector_store = VectorStore(
                chroma_path=temp_dir, embedding_model="all-MiniLM-L6-v2", max_results=5
            )

            assert vector_store.max_results == 5
            assert vector_store.client is not None
            assert vector_store.course_catalog is not None
            assert vector_store.course_content is not None

        finally:
            cleanup_test_files(temp_dir)

    def test_collections_created(self):
        """Test that both collections are created"""
        temp_dir = tempfile.mkdtemp()

        try:
            vector_store = VectorStore(
                chroma_path=temp_dir, embedding_model="all-MiniLM-L6-v2", max_results=3
            )

            # Check collections exist
            collections = vector_store.client.list_collections()
            collection_names = [col.name for col in collections]

            assert "course_catalog" in collection_names
            assert "course_content" in collection_names

        finally:
            cleanup_test_files(temp_dir)


class TestVectorStoreDataOperations:
    """Test adding and retrieving data from VectorStore"""

    def test_add_course_metadata(self):
        """Test adding course metadata to catalog"""
        temp_dir = tempfile.mkdtemp()

        try:
            vector_store = VectorStore(
                chroma_path=temp_dir, embedding_model="all-MiniLM-L6-v2", max_results=5
            )

            # Create test course
            course = create_mock_course(SAMPLE_COURSES[0])

            # Add course metadata
            vector_store.add_course_metadata(course)

            # Verify course was added
            existing_titles = vector_store.get_existing_course_titles()
            assert course.title in existing_titles

            # Verify metadata was stored correctly
            results = vector_store.course_catalog.get(ids=[course.title])
            assert len(results["ids"]) == 1
            assert results["ids"][0] == course.title

            metadata = results["metadatas"][0]
            assert metadata["title"] == course.title
            assert metadata["instructor"] == course.instructor
            assert metadata["course_link"] == course.course_link
            assert metadata["lesson_count"] == len(course.lessons)

            # Check lessons JSON
            lessons = json.loads(metadata["lessons_json"])
            assert len(lessons) == len(course.lessons)
            assert lessons[0]["lesson_number"] == 1

        finally:
            cleanup_test_files(temp_dir)

    def test_add_course_content(self):
        """Test adding course content chunks"""
        temp_dir = tempfile.mkdtemp()

        try:
            vector_store = VectorStore(
                chroma_path=temp_dir, embedding_model="all-MiniLM-L6-v2", max_results=5
            )

            # Create test chunks
            chunks = [
                CourseChunk(
                    course_title="Test Course",
                    lesson_number=1,
                    chunk_index=0,
                    content="This is test content for lesson 1.",
                ),
                CourseChunk(
                    course_title="Test Course",
                    lesson_number=2,
                    chunk_index=1,
                    content="This is test content for lesson 2.",
                ),
            ]

            # Add content chunks
            vector_store.add_course_content(chunks)

            # Verify chunks were added
            results = vector_store.course_content.get()
            assert len(results["ids"]) == 2

            # Check metadata
            for i, metadata in enumerate(results["metadatas"]):
                assert metadata["course_title"] == "Test Course"
                assert metadata["lesson_number"] in [1, 2]
                assert metadata["chunk_index"] in [0, 1]

        finally:
            cleanup_test_files(temp_dir)

    def test_clear_all_data(self):
        """Test clearing all data from vector store"""
        temp_dir = tempfile.mkdtemp()

        try:
            vector_store = VectorStore(
                chroma_path=temp_dir, embedding_model="all-MiniLM-L6-v2", max_results=5
            )

            # Add some test data
            course = create_mock_course(SAMPLE_COURSES[0])
            vector_store.add_course_metadata(course)

            chunks = [CourseChunk("Test", 1, 0, "content")]
            vector_store.add_course_content(chunks)

            # Verify data exists
            assert len(vector_store.get_existing_course_titles()) > 0

            # Clear data
            vector_store.clear_all_data()

            # Verify data is cleared
            assert len(vector_store.get_existing_course_titles()) == 0

        finally:
            cleanup_test_files(temp_dir)


class TestVectorStoreSearch:
    """Test search functionality"""

    def test_search_without_filters(self):
        """Test basic search without course or lesson filters"""
        temp_dir = tempfile.mkdtemp()

        try:
            vector_store = VectorStore(
                chroma_path=temp_dir, embedding_model="all-MiniLM-L6-v2", max_results=3
            )

            # Add test data
            course = create_mock_course(SAMPLE_COURSES[0])
            vector_store.add_course_metadata(course)

            chunks = [
                CourseChunk(
                    "Introduction to Machine Learning",
                    1,
                    0,
                    "Machine learning is a subset of AI",
                ),
                CourseChunk(
                    "Introduction to Machine Learning",
                    2,
                    1,
                    "Supervised learning uses labeled data",
                ),
            ]
            vector_store.add_course_content(chunks)

            # Test search
            results = vector_store.search("machine learning")

            assert results is not None
            assert results.error is None
            assert len(results.documents) > 0
            assert isinstance(results.metadata, list)

        finally:
            cleanup_test_files(temp_dir)

    def test_search_with_course_filter(self):
        """Test search with course name filter"""
        temp_dir = tempfile.mkdtemp()

        try:
            vector_store = VectorStore(
                chroma_path=temp_dir, embedding_model="all-MiniLM-L6-v2", max_results=3
            )

            # Add multiple courses
            course1 = create_mock_course(SAMPLE_COURSES[0])  # ML course
            course2 = create_mock_course(SAMPLE_COURSES[1])  # MCP course

            vector_store.add_course_metadata(course1)
            vector_store.add_course_metadata(course2)

            chunks = [
                CourseChunk(
                    "Introduction to Machine Learning", 1, 0, "ML content here"
                ),
                CourseChunk("Building with MCP", 1, 1, "MCP content here"),
            ]
            vector_store.add_course_content(chunks)

            # Test search with course filter
            results = vector_store.search("content", course_name="Machine Learning")

            assert results.error is None
            if not results.is_empty():
                # Should only return results from ML course
                for metadata in results.metadata:
                    assert "Machine Learning" in metadata.get("course_title", "")

        finally:
            cleanup_test_files(temp_dir)

    def test_search_with_lesson_filter(self):
        """Test search with lesson number filter"""
        temp_dir = tempfile.mkdtemp()

        try:
            vector_store = VectorStore(
                chroma_path=temp_dir, embedding_model="all-MiniLM-L6-v2", max_results=3
            )

            # Add test data with multiple lessons
            course = create_mock_course(SAMPLE_COURSES[0])
            vector_store.add_course_metadata(course)

            chunks = [
                CourseChunk(
                    "Introduction to Machine Learning", 1, 0, "Lesson 1 content"
                ),
                CourseChunk(
                    "Introduction to Machine Learning", 2, 1, "Lesson 2 content"
                ),
                CourseChunk(
                    "Introduction to Machine Learning", 3, 2, "Lesson 3 content"
                ),
            ]
            vector_store.add_course_content(chunks)

            # Test search with lesson filter
            results = vector_store.search("content", lesson_number=2)

            assert results.error is None
            if not results.is_empty():
                # Should only return results from lesson 2
                for metadata in results.metadata:
                    assert metadata.get("lesson_number") == 2

        finally:
            cleanup_test_files(temp_dir)

    def test_search_nonexistent_course(self):
        """Test search for non-existent course"""
        temp_dir = tempfile.mkdtemp()

        try:
            vector_store = VectorStore(
                chroma_path=temp_dir, embedding_model="all-MiniLM-L6-v2", max_results=3
            )

            # Search for course that doesn't exist
            results = vector_store.search("anything", course_name="Nonexistent Course")

            assert results.error is not None
            assert "No course found matching" in results.error

        finally:
            cleanup_test_files(temp_dir)

    def test_search_empty_database(self):
        """Test search on empty database"""
        temp_dir = tempfile.mkdtemp()

        try:
            vector_store = VectorStore(
                chroma_path=temp_dir, embedding_model="all-MiniLM-L6-v2", max_results=3
            )

            # Search on empty database
            results = vector_store.search("anything")

            # Should return empty results, not error
            assert results.error is None
            assert results.is_empty()

        finally:
            cleanup_test_files(temp_dir)


class TestVectorStoreCourseResolution:
    """Test course name resolution functionality"""

    def test_resolve_course_name_exact_match(self):
        """Test resolving exact course name match"""
        temp_dir = tempfile.mkdtemp()

        try:
            vector_store = VectorStore(
                chroma_path=temp_dir, embedding_model="all-MiniLM-L6-v2", max_results=3
            )

            # Add course
            course = create_mock_course(SAMPLE_COURSES[0])
            vector_store.add_course_metadata(course)

            # Test exact match
            resolved = vector_store._resolve_course_name(
                "Introduction to Machine Learning"
            )
            assert resolved == "Introduction to Machine Learning"

        finally:
            cleanup_test_files(temp_dir)

    def test_resolve_course_name_partial_match(self):
        """Test resolving partial course name match"""
        temp_dir = tempfile.mkdtemp()

        try:
            vector_store = VectorStore(
                chroma_path=temp_dir, embedding_model="all-MiniLM-L6-v2", max_results=3
            )

            # Add course
            course = create_mock_course(SAMPLE_COURSES[0])
            vector_store.add_course_metadata(course)

            # Test partial matches
            resolved = vector_store._resolve_course_name("Machine Learning")
            assert resolved == "Introduction to Machine Learning"

            resolved = vector_store._resolve_course_name("ML")
            # This may or may not match depending on embeddings
            print(f"ML resolved to: {resolved}")

        finally:
            cleanup_test_files(temp_dir)

    def test_resolve_course_name_no_match(self):
        """Test resolving non-existent course name"""
        temp_dir = tempfile.mkdtemp()

        try:
            vector_store = VectorStore(
                chroma_path=temp_dir, embedding_model="all-MiniLM-L6-v2", max_results=3
            )

            # No courses added
            resolved = vector_store._resolve_course_name("Nonexistent Course")
            assert resolved is None

        finally:
            cleanup_test_files(temp_dir)


class TestVectorStoreUtilities:
    """Test utility functions"""

    def test_get_course_count(self):
        """Test getting course count"""
        temp_dir = tempfile.mkdtemp()

        try:
            vector_store = VectorStore(
                chroma_path=temp_dir, embedding_model="all-MiniLM-L6-v2", max_results=3
            )

            # Initially should be 0
            assert vector_store.get_course_count() == 0

            # Add courses
            for course_data in SAMPLE_COURSES:
                course = create_mock_course(course_data)
                vector_store.add_course_metadata(course)

            # Should now be 2
            assert vector_store.get_course_count() == len(SAMPLE_COURSES)

        finally:
            cleanup_test_files(temp_dir)

    def test_get_all_courses_metadata(self):
        """Test getting all courses metadata"""
        temp_dir = tempfile.mkdtemp()

        try:
            vector_store = VectorStore(
                chroma_path=temp_dir, embedding_model="all-MiniLM-L6-v2", max_results=3
            )

            # Add courses
            for course_data in SAMPLE_COURSES:
                course = create_mock_course(course_data)
                vector_store.add_course_metadata(course)

            # Get metadata
            all_metadata = vector_store.get_all_courses_metadata()

            assert len(all_metadata) == len(SAMPLE_COURSES)

            for metadata in all_metadata:
                assert "title" in metadata
                assert "instructor" in metadata
                assert "course_link" in metadata
                assert "lessons" in metadata  # Should be parsed from JSON
                assert isinstance(metadata["lessons"], list)

        finally:
            cleanup_test_files(temp_dir)

    def test_get_lesson_link(self):
        """Test getting specific lesson link"""
        temp_dir = tempfile.mkdtemp()

        try:
            vector_store = VectorStore(
                chroma_path=temp_dir, embedding_model="all-MiniLM-L6-v2", max_results=3
            )

            # Add course
            course = create_mock_course(SAMPLE_COURSES[0])
            vector_store.add_course_metadata(course)

            # Get lesson link
            lesson_link = vector_store.get_lesson_link(
                "Introduction to Machine Learning", 1
            )

            assert lesson_link is not None
            assert "lesson1" in lesson_link

            # Test non-existent lesson
            no_link = vector_store.get_lesson_link(
                "Introduction to Machine Learning", 99
            )
            assert no_link is None

        finally:
            cleanup_test_files(temp_dir)


class TestVectorStoreErrorHandling:
    """Test error handling scenarios"""

    def test_add_empty_chunks(self):
        """Test adding empty chunks list"""
        temp_dir = tempfile.mkdtemp()

        try:
            vector_store = VectorStore(
                chroma_path=temp_dir, embedding_model="all-MiniLM-L6-v2", max_results=3
            )

            # Should handle empty list gracefully
            vector_store.add_course_content([])

            # No error should be raised
            assert True

        finally:
            cleanup_test_files(temp_dir)

    def test_invalid_chroma_path(self):
        """Test with invalid ChromaDB path"""
        # This should still work as ChromaDB will create directories
        invalid_path = "/tmp/test_chroma_db_" + str(hash("test"))

        try:
            vector_store = VectorStore(
                chroma_path=invalid_path,
                embedding_model="all-MiniLM-L6-v2",
                max_results=3,
            )

            assert vector_store.client is not None

        finally:
            cleanup_test_files(invalid_path)

    def test_search_with_exception(self):
        """Test search error handling"""
        temp_dir = tempfile.mkdtemp()

        try:
            vector_store = VectorStore(
                chroma_path=temp_dir, embedding_model="all-MiniLM-L6-v2", max_results=3
            )

            # Mock the course_content collection to raise exception
            with patch.object(vector_store.course_content, "query") as mock_query:
                mock_query.side_effect = Exception("Database error")

                results = vector_store.search("test query")

                assert results.error is not None
                assert "Search error" in results.error
                assert "Database error" in results.error

        finally:
            cleanup_test_files(temp_dir)


class TestVectorStoreDiagnosis:
    """Diagnostic tests to identify vector store issues"""

    def test_diagnose_vector_store_state(self):
        """Comprehensive diagnostic test for vector store"""
        temp_dir = tempfile.mkdtemp()

        try:
            print("\\n=== VECTOR STORE DIAGNOSIS ===")

            vector_store = VectorStore(
                chroma_path=temp_dir, embedding_model="all-MiniLM-L6-v2", max_results=3
            )

            print(f"1. Vector store initialized: {vector_store is not None}")
            print(f"2. Client exists: {vector_store.client is not None}")
            print(
                f"3. Course catalog collection: {vector_store.course_catalog is not None}"
            )
            print(
                f"4. Course content collection: {vector_store.course_content is not None}"
            )

            # Test collections
            try:
                collections = vector_store.client.list_collections()
                print(f"5. Collections count: {len(collections)}")
                for col in collections:
                    print(f"   - {col.name}")
            except Exception as e:
                print(f"5. Collections error: {e}")

            # Test empty search
            try:
                empty_results = vector_store.search("anything")
                print(f"6. Empty search error: {empty_results.error}")
                print(f"   Empty search documents: {len(empty_results.documents)}")
            except Exception as e:
                print(f"6. Empty search exception: {e}")

            # Add test data
            print("7. Adding test data...")
            try:
                course = create_mock_course(SAMPLE_COURSES[0])
                vector_store.add_course_metadata(course)

                chunks = [
                    CourseChunk(
                        "Test Course", 1, 0, "Test content about machine learning"
                    )
                ]
                vector_store.add_course_content(chunks)

                print("   Data added successfully")
            except Exception as e:
                print(f"   Data addition error: {e}")

            # Test search with data
            try:
                search_results = vector_store.search("machine learning")
                print(f"8. Search with data error: {search_results.error}")
                print(f"   Search documents: {len(search_results.documents)}")
                if search_results.documents:
                    print(f"   First document: {search_results.documents[0][:50]}...")
            except Exception as e:
                print(f"8. Search with data exception: {e}")

            print("=== END VECTOR STORE DIAGNOSIS ===\\n")

        finally:
            cleanup_test_files(temp_dir)
