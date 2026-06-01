# Student Performance Analysis Report

This report presents exploratory data analysis and key insights from the student performance dataset.

## 1. Key Performance Indicators (KPIs)
- **Total Students Analyzed**: 1000
- **Average Attendance**: 81.42%
- **Average Study Hours/Day**: 5.06 hrs
- **Average Sleep Hours/Night**: 7.02 hrs
- **Average Previous Grade**: 82.86/100
- **Average Final Exam Marks**: 58.50/100
- **Overall Pass Rate**: 78.60% (786 passed, 214 failed)

## 2. Impact of Study Hours on Final Marks
| Study Category | Average Final Marks | Student Count |
|---|---|---|
| Low (<3 hrs) | 47.46 | 162 |
| Medium (3-6 hrs) | 56.13 | 507 |
| High (>6 hrs) | 67.54 | 331 |

## 3. Impact of Attendance on Final Marks
| Attendance Category | Average Final Marks | Student Count |
|---|---|---|
| Poor (<75%) | 55.48 | 260 |
| Average (75-90%) | 58.93 | 537 |
| Excellent (90-100%) | 61.23 | 203 |

## 4. Impact of Sleep on Academic Results
| Sleep Category | Average Final Marks | Student Count |
|---|---|---|
| Insufficient (<6 hrs) | 55.79 | 200 |
| Healthy (6-8.5 hrs) | 59.46 | 680 |
| Excessive (>8.5 hrs) | 57.61 | 120 |

## 5. Demographic and Department Comparison
### Performance by Gender
| Gender | Average Marks | Pass Rate (%) |
|---|---|---|
| Female | 57.85 | 78.79% |
| Male | 59.19 | 78.40% |

### Performance by Department
| Department | Average Marks | Pass Rate (%) |
|---|---|---|
| Business Administration | 58.44 | 80.48% |
| Computer Science | 58.51 | 80.68% |
| Electrical Engineering | 58.19 | 75.46% |
| Information Technology | 58.83 | 79.17% |
| Mechanical Engineering | 58.58 | 77.67% |

## 6. Correlation Analysis
Correlation values with Final_Exam_Marks:
| Variable | Correlation with Final Marks |
|---|---|
| Study_Hours | 0.7162 |
| Previous_Grades | 0.4349 |
| Attendance_Percentage | 0.2312 |
| Sleep_Hours | 0.0974 |
| Internet_Usage_Hours | 0.0074 |

## 7. Top 10 Performing Students
| Student ID | Name | Department | Attendance | Study Hours | Final Marks |
|---|---|---|---|---|---|
| STU0607 | Charles Moore | Electrical Engineering | 89.8% | 8.1 | 89.0 |
| STU0387 | Steven Johnson | Electrical Engineering | 93.3% | 6.1 | 89.0 |
| STU0029 | Karen Lopez | Mechanical Engineering | 82.0% | 10.0 | 86.5 |
| STU0005 | David Davis | Computer Science | 97.7% | 9.3 | 86.5 |
| STU0017 | Robert Moore | Business Administration | 88.3% | 8.5 | 86.3 |
| STU0697 | William Thomas | Mechanical Engineering | 97.8% | 7.9 | 85.1 |
| STU0830 | Steven Anderson | Electrical Engineering | 97.7% | 8.2 | 85.1 |
| STU0303 | Donna Gonzalez | Business Administration | 83.5% | 8.5 | 84.3 |
| STU0993 | Joshua Anderson | Electrical Engineering | 83.3% | 7.7 | 84.1 |
| STU0510 | Linda Hernandez | Computer Science | 91.8% | 4.9 | 83.0 |

## 8. Bottom 10 Performing Students
| Student ID | Name | Department | Attendance | Study Hours | Final Marks |
|---|---|---|---|---|---|
| STU0854 | Lisa Smith | Electrical Engineering | 65.3% | 3.3 | 26.0 |
| STU0201 | Betty Jackson | Electrical Engineering | 70.9% | 1.7 | 26.1 |
| STU0244 | Robert Wilson | Computer Science | 76.3% | 2.7 | 28.1 |
| STU0234 | Margaret Wilson | Electrical Engineering | 72.2% | 4.5 | 28.5 |
| STU0030 | John Williams | Business Administration | 74.8% | 1.8 | 29.6 |
| STU0424 | Margaret Jones | Electrical Engineering | 74.9% | 2.0 | 29.9 |
| STU0835 | Elizabeth Jackson | Information Technology | 62.2% | 2.5 | 29.9 |
| STU0006 | Paul Lopez | Electrical Engineering | 65.9% | 1.0 | 32.5 |
| STU0733 | Mary Davis | Business Administration | 79.3% | 2.1 | 32.7 |
| STU0036 | Nancy Thomas | Business Administration | 59.6% | 3.7 | 32.9 |


## 9. Machine Learning Predictive Analysis
We trained three regression models to predict final exam marks based on attendance, study hours, sleep hours, and previous grades.

| Model Name | Mean Absolute Error (MAE) | Root Mean Squared Error (RMSE) | R-squared (R2) Score |
|---|---|---|---|
| Linear Regression | 4.6463 | 5.7330 | 0.6763 |
| Decision Tree | 5.0914 | 6.4960 | 0.5844 |
| Random Forest | 4.6256 | 5.8708 | 0.6606 |

**Conclusion**: The **Linear Regression** model outperformed the other models with a R-squared score of **67.6315%**, and has been exported for integration into the interactive dashboard.
