/*
 * Copyright 2025, Sirius Contributors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "gpu_physical_plan_generator.hpp"
#include "gpu_physical_grouped_aggregate.hpp"
#include "gpu_physical_projection.hpp"
#include "duckdb/function/aggregate/distributive_function_utils.hpp"
#include "duckdb/function/function_binder.hpp"
#include "duckdb/main/client_config.hpp"
#include "duckdb/optimizer/rule/ordered_aggregate_optimizer.hpp"
#include "duckdb/planner/expression/bound_aggregate_expression.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/planner/operator/logical_distinct.hpp"

namespace duckdb {

unique_ptr<GPUPhysicalOperator> GPUPhysicalPlanGenerator::CreatePlan(LogicalDistinct &op) {
    D_ASSERT(op.children.size() == 1);
    auto child = CreatePlan(*op.children[0]);
    auto &distinct_targets = op.distinct_targets;
    D_ASSERT(child);
    D_ASSERT(!distinct_targets.empty());

    auto &types = child->GetTypes();
    vector<unique_ptr<Expression>> groups;
    vector<unique_ptr<Expression>> aggregates;
    vector<unique_ptr<Expression>> projections;
    idx_t group_count = distinct_targets.size();
    unordered_map<idx_t, idx_t> group_by_references;
    vector<LogicalType> aggregate_types;

    for (idx_t i = 0; i < distinct_targets.size(); i++) {
        auto &target = distinct_targets[i];
        if (target->GetExpressionType() == ExpressionType::BOUND_REF) {
            auto &bound_ref = target->Cast<BoundReferenceExpression>();
            group_by_references[bound_ref.index] = i;
        }
        aggregate_types.push_back(target->return_type);
        groups.push_back(std::move(target));
    }

    bool requires_projection = types.size() != group_count;

    for (idx_t i = 0; i < types.size(); ++i) {
        auto logical_type = types[i];
        auto entry = group_by_references.find(i);
        if (entry != group_by_references.end()) {
            auto group_index = entry->second;
            projections.push_back(make_uniq<BoundReferenceExpression>(logical_type, group_index));
            if (group_index != i) {
                requires_projection = true;
            }
        } else {
            if (op.distinct_type == DistinctType::DISTINCT && op.order_by) {
                throw InternalException("Entry that is not a group, but not a DISTINCT ON aggregate");
            }
            auto bound = make_uniq<BoundReferenceExpression>(logical_type, i);
            vector<unique_ptr<Expression>> first_children;
            first_children.push_back(std::move(bound));

            FunctionBinder function_binder(context);
            auto first_aggregate = function_binder.BindAggregateFunction(
                FirstFunctionGetter::GetFunction(logical_type), std::move(first_children), nullptr,
                AggregateType::NON_DISTINCT);
            first_aggregate->order_bys = op.order_by ? op.order_by->Copy() : nullptr;

            if (ClientConfig::GetConfig(context).enable_optimizer) {
                bool changes_made = false;
                auto new_expr = OrderedAggregateOptimizer::Apply(context, *first_aggregate, groups, changes_made);
                if (new_expr) {
                    D_ASSERT(new_expr->return_type == first_aggregate->return_type);
                    D_ASSERT(new_expr->GetExpressionType() == ExpressionType::BOUND_AGGREGATE);
                    first_aggregate = unique_ptr_cast<Expression, BoundAggregateExpression>(std::move(new_expr));
                }
            }
            projections.push_back(
                make_uniq<BoundReferenceExpression>(logical_type, group_count + aggregates.size()));
            aggregate_types.push_back(logical_type);
            aggregates.push_back(std::move(first_aggregate));
            requires_projection = true;
        }
    }

    child = ExtractAggregateExpressions(std::move(child), aggregates, groups);

    auto groupby = make_uniq_base<GPUPhysicalOperator, GPUPhysicalGroupedAggregate>(
        context, aggregate_types, std::move(aggregates), std::move(groups), child->estimated_cardinality);
    groupby->children.push_back(std::move(child));

    if (!requires_projection) {
        return groupby;
    }

    auto aggr_projection =
        make_uniq<GPUPhysicalProjection>(types, std::move(projections), groupby->estimated_cardinality);
    aggr_projection->children.push_back(std::move(groupby));
    return aggr_projection;
}

} // namespace duckdb
